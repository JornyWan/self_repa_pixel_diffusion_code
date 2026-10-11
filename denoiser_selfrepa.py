"""JiT + Self-Distillation Representation Alignment (SELF-CONTAINED; NO external DINOv2).

Ports SRA (Jiang et al. 2025, latent) into pixel-space JiT: an EARLY block's patch tokens of
the ONLINE net (from the noisier training input z_t) are projected and cosine-aligned to a LATE
block's patch tokens of an EMA-TEACHER (from a CLEANER version of the same image). No pretrained
teacher -> keeps JiT's self-contained thesis, unlike PixelREPA/PixelDiT/RAE which all need DINOv2.

Anti-collapse (BYOL-style, same as SRA): stop-grad on the teacher (EMA, no_grad) + asymmetry in
layer (early->late) and noise (student noisier than teacher) + an online predictor (projector).
Teacher + projector are TRAINING-ONLY (discarded at inference; generate() is inherited)."""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from denoiser import Denoiser


class DenoiserSelfREPA(Denoiser):
    def __init__(self, args):
        super().__init__(args)
        self.s_block = args.self_student_block        # early block on ONLINE net (noisier)
        self.t_block = args.self_teacher_block        # late  block on EMA teacher (cleaner)
        self.self_lambda = args.self_lambda
        self.teacher_ema_decay = args.self_teacher_ema
        self.teacher_dt = args.self_teacher_dt        # fraction of noise removed for the teacher view
        # iREPA-style spatial normalization (opt-in): removes the shared/global per-channel component
        # (e.g. the fixed sincos pos_embed) so alignment targets SPATIAL STRUCTURE, not a trivial
        # global component. Off by default -> a resumed base pilot is byte-for-byte unaffected.
        self.spatial_norm = getattr(args, 'self_spatial_norm', False)
        self.spatial_gamma = getattr(args, 'self_spatial_gamma', 1.0)
        # noise-level alignment weighting w(t): 'uniform' (=old behavior) or 'bandpass' (up-weight mid-noise where
        # alignment is well-conditioned, down-weight the ill-conditioned high-noise & the not-needed near-clean ends).
        self.lam_schedule = getattr(args, 'self_lam_schedule', 'uniform')
        self.lam_tstar = getattr(args, 'self_lam_tstar', 0.5)
        self.lam_sigma = getattr(args, 'self_lam_sigma', 0.25)
        # v4 (SRA-faithful) knobs, all default to old behavior:
        #  dt_mode='uniform' -> teacher offset is a small RANDOM additive k~U[0,dt_max] (SRA recipe) instead of fixed.
        #  loss='smoothl1'   -> smooth-L1 on per-token standardized features (SRA) instead of neg-cosine.
        #  structural        -> add patch-to-patch similarity-MATRIX matching (targets SPATIAL STRUCTURE, 2512.10794).
        #  lambda_decay      -> SRA's late-training lambda decay (near-inert before decay_start).
        self.dt_mode = getattr(args, 'self_dt_mode', 'fixed')
        self.dt_max = getattr(args, 'self_dt_max', 0.2)
        self.loss_type = getattr(args, 'self_loss', 'cosine')
        self.structural = getattr(args, 'self_structural', False)
        self.struct_lambda = getattr(args, 'self_struct_lambda', 1.0)
        self.lambda_decay = getattr(args, 'self_lambda_decay', False)
        self.decay_start = getattr(args, 'self_lambda_decay_start', 150)
        self.cur_epoch = 0
        hid = self.net.hidden_size

        # EMA teacher = frozen copy of the net (synced in update_ema); training-only, discarded at inference
        self.teacher = copy.deepcopy(self.net)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # BYOL-style online predictor: student(hid) -> teacher feature space(hid). Discarded at inference.
        self.projector = nn.Sequential(
            nn.Linear(hid, 2048), nn.SiLU(), nn.Linear(2048, 2048), nn.SiLU(), nn.Linear(2048, hid))

        self._s_feat = None
        self._t_feat = None
        self.net.blocks[self.s_block].register_forward_hook(lambda m, i, o: setattr(self, "_s_feat", o))
        self.teacher.blocks[self.t_block].register_forward_hook(lambda m, i, o: setattr(self, "_t_feat", o))

    @torch.no_grad()
    def update_ema(self):
        super().update_ema()                          # JiT's ema_params1/2 (for sampling) -- unchanged
        d = self.teacher_ema_decay                    # teacher <- EMA(online net); stays in DDP sync
        for pt, ps in zip(self.teacher.parameters(), self.net.parameters()):
            pt.mul_(d).add_(ps.detach(), alpha=1 - d)
        for bt, bs in zip(self.teacher.buffers(), self.net.buffers()):
            bt.copy_(bs)

    def _spatial_norm(self, x, eps=1e-6):
        # normalize each channel across the token (spatial) dimension: y=(x - gamma*mean_tok)/std_tok
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - self.spatial_gamma * mu) / (var + eps).sqrt()

    def _std_norm(self, x, eps=1e-5):
        # per-token standardization across feature dim (for smooth-L1 regression to a scale-free target)
        return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + eps)

    def forward(self, x, labels):
        labels_dropped = self.drop_labels(labels) if self.training else labels
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale
        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)
        x_pred = self.net(z, t.flatten(), labels_dropped)          # hook -> _s_feat (student, early block)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
        diff_loss = ((v - v_pred) ** 2).mean(dim=(1, 2, 3)).mean()

        # teacher sees a CLEANER version of the SAME sample (less noise), read at a LATER block
        if self.dt_mode == 'uniform':                              # SRA recipe: small RANDOM additive offset k~U[0,dt_max]
            k = torch.rand(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1))) * self.dt_max
            t2 = (t + k).clamp(max=1.0)
        else:
            t2 = t + self.teacher_dt * (1 - t)                     # (fixed mode) move a fraction dt toward clean
        z2 = t2 * x + (1 - t2) * e
        with torch.no_grad():
            self.teacher(z2, t2.flatten(), labels_dropped)         # hook -> _t_feat (teacher, late block)

        s = self.projector(self._s_feat[:, self.net.in_context_len:])   # [N,256,hid] student patch tokens
        tgt = self._t_feat[:, self.net.in_context_len:].detach()        # [N,256,hid] teacher patch tokens
        if self.spatial_norm:                                           # iREPA: strip shared global per-channel comp
            s = self._spatial_norm(s)
            tgt = self._spatial_norm(tgt.float()).to(s.dtype)
        # --- point-wise alignment term (per-sample [N], lower=better) ---
        if self.loss_type == 'smoothl1':                            # SRA: smooth-L1 to standardized teacher features
            sn = self._std_norm(s)
            tn = self._std_norm(tgt.float()).to(s.dtype)
            align_ps = F.smooth_l1_loss(sn, tn, reduction='none').mean(-1).mean(dim=1)
        else:                                                       # REPA/SRA-lite: negative cosine
            sn = F.normalize(s, dim=-1)
            tn = F.normalize(tgt.float(), dim=-1).to(s.dtype)
            align_ps = -(sn * tn).sum(-1).mean(dim=1)
        # --- structural term: match patch-to-patch cosine-sim MATRIX (spatial structure, 2512.10794) ---
        if self.structural:
            Sn = F.normalize(s, dim=-1)
            Tn = F.normalize(tgt.float(), dim=-1).to(s.dtype)
            Gs = torch.bmm(Sn, Sn.transpose(1, 2))
            Gt = torch.bmm(Tn, Tn.transpose(1, 2)).detach()
            struct_ps = (Gs - Gt).pow(2).mean(dim=(1, 2))           # per-sample [N]
            self.last_struct = float(struct_ps.mean().detach())
            align_ps = align_ps + self.struct_lambda * struct_ps
        if self.lam_schedule == 'bandpass':                         # weight by noise level; normalize to mean 1 (fair)
            tt = t.flatten()
            w = torch.exp(-((tt - self.lam_tstar) ** 2) / (2 * self.lam_sigma ** 2))
            w = (w / w.mean().clamp_min(1e-6)).detach()
            align_loss = (w * align_ps).mean()
        else:
            align_loss = align_ps.mean()
        lam = self.self_lambda                                      # SRA late-training lambda decay (near-inert early)
        if self.lambda_decay and self.cur_epoch > self.decay_start:
            lam = self.self_lambda * (0.1 ** ((self.cur_epoch - self.decay_start) / 1000.0))

        self.last_diff = float(diff_loss.detach())
        self.last_repa = float(align_loss.detach())
        return diff_loss + lam * align_loss
