"""JiT + Self-Distillation Representation Alignment full training (SELF-CONTAINED, no DINOv2).
= main_jit.py with DenoiserSelfREPA. Comparable protocol: ImageNet-256, FID-50k vs
jit_in256_stats.npz, cfg-guided online eval. Baseline = vanilla JiT-B/16 (official 3.64/3.66).
Key test: does self-alignment beat vanilla at convergence WITHOUT the cross-over that external
REPA suffers on JiT (PixelREPA 2603.14366)?"""
import argparse, datetime, numpy as np, os, time, copy
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from util.crop import center_crop_arr
import util.misc as misc
from engine_jit import train_one_epoch, evaluate
from denoiser_selfrepa import DenoiserSelfREPA


def get_args_parser():
    p = argparse.ArgumentParser('JiT-SelfREPA', add_help=False)
    p.add_argument('--model', default='JiT-B/16', type=str)
    p.add_argument('--img_size', default=256, type=int)
    p.add_argument('--attn_dropout', type=float, default=0.0)
    p.add_argument('--proj_dropout', type=float, default=0.0)
    # --- self-distillation alignment (self-contained) ---
    p.add_argument('--self_student_block', type=int, default=4)   # early block on ONLINE net (noisier)
    p.add_argument('--self_teacher_block', type=int, default=8)   # late  block on EMA teacher (cleaner)
    p.add_argument('--self_lambda', type=float, default=0.5)      # alignment weight
    p.add_argument('--self_teacher_ema', type=float, default=0.9999)  # teacher EMA decay (SRA-style)
    p.add_argument('--self_teacher_dt', type=float, default=0.5)  # fraction of noise removed for teacher view
    p.add_argument('--self_spatial_norm', action='store_true')    # iREPA spatial norm (strip shared global comp)
    p.add_argument('--self_spatial_gamma', type=float, default=1.0)
    p.add_argument('--self_lam_schedule', type=str, default='uniform', choices=['uniform', 'bandpass'])  # w(t) on alignment
    p.add_argument('--self_lam_tstar', type=float, default=0.5)    # band-pass center (mid-noise)
    p.add_argument('--self_lam_sigma', type=float, default=0.25)   # band-pass width
    # v4 (SRA-faithful) knobs:
    p.add_argument('--self_dt_mode', type=str, default='fixed', choices=['fixed', 'uniform'])  # uniform=random k~U[0,dt_max]
    p.add_argument('--self_dt_max', type=float, default=0.2)       # upper bound of random teacher offset (SRA: 0.2)
    p.add_argument('--self_loss', type=str, default='cosine', choices=['cosine', 'smoothl1'])  # SRA uses smooth-L1
    p.add_argument('--self_structural', action='store_true')       # add patch-sim-matrix (spatial-structure) term
    p.add_argument('--self_struct_lambda', type=float, default=1.0)
    p.add_argument('--self_lambda_decay', action='store_true')     # SRA late-training lambda decay
    p.add_argument('--self_lambda_decay_start', type=int, default=150)
    p.add_argument('--epochs', default=600, type=int)
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--batch_size', default=128, type=int)
    p.add_argument('--accum_iter', default=1, type=int)   # gradient accumulation: eff_batch = batch_size*world*accum
    p.add_argument('--max_steps', default=0, type=int)    # >0: stop epoch early (smoke only)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--blr', type=float, default=5e-5)
    p.add_argument('--min_lr', type=float, default=0.)
    p.add_argument('--lr_schedule', type=str, default='constant')
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--ema_decay1', type=float, default=0.9999)
    p.add_argument('--ema_decay2', type=float, default=0.9996)
    p.add_argument('--P_mean', default=-0.8, type=float)
    p.add_argument('--P_std', default=0.8, type=float)
    p.add_argument('--noise_scale', default=1.0, type=float)
    p.add_argument('--t_eps', default=5e-2, type=float)
    p.add_argument('--label_drop_prob', default=0.1, type=float)
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--start_epoch', default=0, type=int)
    p.add_argument('--num_workers', default=12, type=int)
    p.add_argument('--pin_mem', action='store_true'); p.set_defaults(pin_mem=True)
    p.add_argument('--sampling_method', default='heun', type=str)
    p.add_argument('--num_sampling_steps', default=50, type=int)
    p.add_argument('--cfg', default=3.0, type=float)
    p.add_argument('--interval_min', default=0.1, type=float)
    p.add_argument('--interval_max', default=1.0, type=float)
    p.add_argument('--num_images', default=50000, type=int)
    p.add_argument('--eval_freq', type=int, default=40)
    p.add_argument('--online_eval', action='store_true')
    p.add_argument('--evaluate_gen', action='store_true')
    p.add_argument('--gen_bsz', type=int, default=256)
    p.add_argument('--data_path', default='./data/imagenet', type=str)
    p.add_argument('--class_num', default=1000, type=int)
    p.add_argument('--output_dir', default='./output_dir')
    p.add_argument('--resume', default='')
    p.add_argument('--save_last_freq', type=int, default=5)
    p.add_argument('--log_freq', default=100, type=int)
    p.add_argument('--device', default='cuda')
    p.add_argument('--world_size', default=1, type=int)
    p.add_argument('--local_rank', default=-1, type=int)
    p.add_argument('--dist_on_itp', action='store_true')
    p.add_argument('--dist_url', default='env://')
    return p


def main(args):
    misc.init_distributed_mode(args)
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed); np.random.seed(seed); cudnn.benchmark = True
    num_tasks = misc.get_world_size(); global_rank = misc.get_rank()
    log_writer = SummaryWriter(log_dir=args.output_dir) if (global_rank == 0 and args.output_dir) else None

    transform_train = transforms.Compose([
        transforms.Lambda(lambda im: center_crop_arr(im, args.img_size)),
        transforms.RandomHorizontalFlip(), transforms.PILToTensor()])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
        persistent_workers=True)  # workers forked once (pre-compile/eval); avoids epoch-boundary refork crash

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    model = DenoiserSelfREPA(args).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_train/1e6:.2f}M (net+projector; EMA teacher frozen)")

    eff_batch_size = args.batch_size * misc.get_world_size() * args.accum_iter
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print(f"lr {args.lr:.2e}  eff_batch {eff_batch_size} (accum_iter {args.accum_iter})")

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
    model_without_ddp = model.module
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    ckpt_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location='cpu')
        model_without_ddp.load_state_dict(ck['model'])
        model_without_ddp.ema_params1 = [ck['model_ema1'][n].cuda() for n, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ck['model_ema2'][n].cuda() for n, _ in model_without_ddp.named_parameters()]
        if 'optimizer' in ck and 'epoch' in ck:
            optimizer.load_state_dict(ck['optimizer']); args.start_epoch = ck['epoch'] + 1
        print("Resumed from", args.resume, "epoch", args.start_epoch); del ck
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Training from scratch")

    if args.evaluate_gen:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate(model_without_ddp, args, 0, batch_size=args.gen_bsz, log_writer=log_writer)
        return

    print(f"Start Self-REPA training {args.epochs} epochs "
          f"(student_block={args.self_student_block} teacher_block={args.self_teacher_block} "
          f"lambda={args.self_lambda} teacher_ema={args.self_teacher_ema} teacher_dt={args.self_teacher_dt})")
    t0 = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        model_without_ddp.cur_epoch = epoch                       # for SRA lambda-decay schedule
        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args)
        print(f"[epoch {epoch}] SelfREPA diff_loss={getattr(model_without_ddp,'last_diff',float('nan')):.4f} "
              f"align_loss={getattr(model_without_ddp,'last_repa',float('nan')):.4f}", flush=True)
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, epoch=epoch, epoch_name="last")
        if epoch % 100 == 0 and epoch > 0:
            misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, epoch=epoch)
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer)
            torch.cuda.empty_cache()
        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()
    print('Training time', str(datetime.timedelta(seconds=int(time.time() - t0))))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
