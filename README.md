# Self-REPA for Pixel-Space Diffusion Transformers (JiT)

Self-contained **representation alignment** for the pixel-space diffusion transformer **JiT**.
An early online denoiser block is aligned — through a small predictor and an **EMA self-teacher**
read at a deeper block on a *cleaner* view of the same image — to a commensurate, same-dimensional
target the model produces for itself. No pretrained encoder, no extra data; the sampler is unchanged.

`L = L_gen + λ · L_align`, with `L_align` a patchwise negative-cosine loss between a projected early
student block and a stop-gradient deeper EMA-teacher block on a lower-noise view (`t⁺ = t + Δ(1−t)`).

## Attribution & license

This repository **builds on** [JiT](https://github.com/LTH14/JiT) (Tianhong Li, *"Back to Basics:
Let Denoising Generative Models Denoise"*, [arXiv:2511.13720](https://arxiv.org/abs/2511.13720)),
which is MIT-licensed — see `LICENSE` (© 2025 Tianhong Li) and the upstream readme `README_JiT.md`.
The base JiT files (`main_jit.py`, `model_jit.py`, `denoiser.py`, `engine_jit.py`, `util/`,
`prepare_ref.py`, `environment.yaml`, `fid_stats/`) are from JiT. The **Self-REPA addition** is:

| file | what it is |
|---|---|
| `denoiser_selfrepa.py` | `DenoiserSelfREPA` — the self-distillation alignment on top of JiT's `Denoiser` |
| `jit_selfrepa_train.py` | training / evaluation entry point (adds `--self_*` flags) |
| `smoke_selfrepa.py` / `.sh` | standalone smoke test (no DDP / no data) |

## Setup

```bash
conda env create -f environment.yaml      # creates env `jit` (python 3.10, torch 2.5.1 / cu124)
conda activate jit
```

Prepare ImageNet (256×256) and point `--data_path` at it. FID/IS use the bundled reference stats in
`fid_stats/` (JiT's `jit_in256_stats.npz`); `prepare_ref.py` can regenerate them if needed.

Quick sanity check (1 GPU, no data):

```bash
bash smoke_selfrepa.sh
```

## Training

Matched protocol: AdamW, **effective batch 1024**, base LR `5e-5` (scaled by batch/256), 5-epoch
warmup then constant, JiT's logit-normal noise. Set `--batch_size` (per-GPU) and `--accum_iter` so
that `batch_size × num_gpus × accum_iter = 1024`. The teacher/predictor are training-only.

**JiT-B/16** (12 blocks; student→teacher block `4→8`):

```bash
torchrun --nproc_per_node=$NGPU --standalone jit_selfrepa_train.py \
    --model JiT-B/16 --img_size 256 --noise_scale 1.0 \
    --self_student_block 4 --self_teacher_block 8 --self_lambda 0.1 \
    --self_teacher_ema 0.9999 --self_teacher_dt 0.5 \
    --batch_size $((1024/NGPU)) --blr 5e-5 --epochs 600 --warmup_epochs 5 \
    --online_eval --eval_freq 40 --save_last_freq 5 \
    --gen_bsz 256 --num_images 50000 --cfg 3.0 --interval_min 0.1 --interval_max 1.0 \
    --data_path /path/to/imagenet_256 --output_dir /path/to/out --resume /path/to/out
```

**JiT-L/16** (24 blocks; student→teacher block `8→16`):

```bash
torchrun --nproc_per_node=$NGPU --standalone jit_selfrepa_train.py \
    --model JiT-L/16 --img_size 256 --noise_scale 1.0 \
    --self_student_block 8 --self_teacher_block 16 --self_lambda 0.1 \
    --self_teacher_ema 0.9999 --self_teacher_dt 0.5 \
    --batch_size 64 --accum_iter $((1024/(64*NGPU))) --blr 5e-5 --epochs 600 --warmup_epochs 5 \
    --online_eval --eval_freq 40 --save_last_freq 5 \
    --gen_bsz 64 --num_images 50000 --cfg 3.0 --interval_min 0.1 --interval_max 1.0 \
    --data_path /path/to/imagenet_256 --output_dir /path/to/out --resume /path/to/out
```

A matched **baseline** (no alignment) is JiT's own `main_jit.py` with the same training args.
`--online_eval` logs FID/IS every `eval_freq` epochs to TensorBoard (tag `fid_cfg3.0_res256`).

## Evaluation (FID/IS, 50k)

Evaluate a trained checkpoint with `--evaluate_gen` at the model's **optimal guidance** (per-size:
JiT-B/16 → CFG **3.0**, JiT-L/16 → CFG **2.6**), with `--resume` pointing at the output dir:

```bash
torchrun --nproc_per_node=2 --standalone jit_selfrepa_train.py \
    --model JiT-L/16 --img_size 256 --noise_scale 1.0 \
    --self_student_block 8 --self_teacher_block 16 \
    --self_lambda 0.1 --self_teacher_ema 0.9999 --self_teacher_dt 0.5 \
    --evaluate_gen --gen_bsz 64 --num_images 50000 \
    --cfg 2.6 --interval_min 0.1 --interval_max 1.0 \
    --data_path /path/to/imagenet_256 --resume /path/to/out --output_dir /path/to/out
```

> Self-REPA checkpoints carry `net.*` + `teacher.*` + `projector.*` keys, so they must be loaded
> through `jit_selfrepa_train.py` (not `main_jit.py`). The `--self_*` flags are construction-only at
> sampling time (the sampler is JiT's original). The baseline arm uses `main_jit.py`.
