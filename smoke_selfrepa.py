"""Standalone smoke for DenoiserSelfREPA (no DDP/data): verify teacher EMA sync, hooks fire,
loss balance, trainable-param exclusion of the frozen teacher, and memory."""
import copy, types, torch
from denoiser_selfrepa import DenoiserSelfREPA

args = types.SimpleNamespace(
    model='JiT-B/16', img_size=256, class_num=1000, attn_dropout=0.0, proj_dropout=0.0,
    label_drop_prob=0.1, P_mean=-0.8, P_std=0.8, t_eps=5e-2, noise_scale=1.0,
    ema_decay1=0.9999, ema_decay2=0.9996, sampling_method='heun', num_sampling_steps=50,
    cfg=3.0, interval_min=0.1, interval_max=1.0,
    self_student_block=4, self_teacher_block=8, self_lambda=0.5,
    self_teacher_ema=0.9999, self_teacher_dt=0.5)

torch.manual_seed(0)
dev = 'cuda'
m = DenoiserSelfREPA(args).to(dev)
m.ema_params1 = copy.deepcopy(list(m.parameters()))
m.ema_params2 = copy.deepcopy(list(m.parameters()))

n_train = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
n_teacher = sum(p.numel() for p in m.teacher.parameters()) / 1e6
n_proj = sum(p.numel() for p in m.projector.parameters()) / 1e6
n_net = sum(p.numel() for p in m.net.parameters()) / 1e6
print(f"net={n_net:.2f}M  projector={n_proj:.2f}M  teacher(frozen)={n_teacher:.2f}M  TRAINABLE={n_train:.2f}M")
frozen_net = [(n, tuple(p.shape)) for n, p in m.net.named_parameters() if not p.requires_grad]
frozen_proj = [(n, tuple(p.shape)) for n, p in m.projector.named_parameters() if not p.requires_grad]
print(f"frozen in net: {frozen_net}")
print(f"frozen in projector: {frozen_proj}")
n_total = sum(p.numel() for p in m.parameters()) / 1e6
n_net_train = sum(p.numel() for p in m.net.parameters() if p.requires_grad) / 1e6
n_proj_train = sum(p.numel() for p in m.projector.parameters() if p.requires_grad) / 1e6
print(f"total={n_total:.2f}M  net_trainable={n_net_train:.2f}M  proj_trainable={n_proj_train:.2f}M")
# correct invariant: teacher contributes ZERO trainable params (net's frozen sincos pos_embed is fine)
assert all(not p.requires_grad for p in m.teacher.parameters()), "teacher must be frozen"
assert abs(n_train - (n_net_train + n_proj_train)) < 0.01, "teacher must add zero trainable params"

opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4, betas=(0.9, 0.95))
m.train(True)
B = 32
# snapshot a teacher param + a net param to verify EMA behavior
t_before = m.teacher.blocks[8].state_dict()[list(m.teacher.blocks[8].state_dict())[0]].clone()
for step in range(4):
    x = (torch.rand(B, 3, 256, 256, device=dev) * 2 - 1)
    labels = torch.randint(0, 1000, (B,), device=dev)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        loss = m(x, labels)
    opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    m.update_ema()
    sfeat_ok = m._s_feat is not None and m._t_feat is not None
    print(f"step {step}: loss={loss.item():.4f}  diff={m.last_diff:.4f}  align={m.last_repa:.4f}  "
          f"s_feat={tuple(m._s_feat.shape)} t_feat={tuple(m._t_feat.shape)}  hooks_ok={sfeat_ok}")
    assert torch.isfinite(loss), "loss must be finite"

t_after = m.teacher.blocks[8].state_dict()[list(m.teacher.blocks[8].state_dict())[0]].clone()
teacher_moved = (t_before - t_after).abs().max().item()
print(f"teacher param moved by EMA: max|delta|={teacher_moved:.3e} (should be >0, small)")
assert teacher_moved > 0, "teacher EMA must update"
# align_loss should be a valid cosine in [-1, 0]; diff should be the usual JiT ~O(1) early
assert -1.01 <= m.last_repa <= 0.01, "align must be negative cosine"
print(f"max mem: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
print("SELFREPA_SMOKE_OK")
