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

**JiT-H/16** (32 blocks; student→teacher block `10→20` — recommended, following the same rule; not yet run by us):

```bash
torchrun --nproc_per_node=$NGPU --standalone jit_selfrepa_train.py \
    --model JiT-H/16 --img_size 256 --noise_scale 1.0 \
    --self_student_block 10 --self_teacher_block 20 --self_lambda 0.1 \
    --self_teacher_ema 0.9999 --self_teacher_dt 0.5 \
    --batch_size 32 --accum_iter $((1024/(32*NGPU))) --blr 5e-5 --epochs 600 --warmup_epochs 5 \
    --online_eval --eval_freq 40 --save_last_freq 5 \
    --gen_bsz 64 --num_images 50000 --cfg 2.2 --interval_min 0.1 --interval_max 1.0 \
    --data_path /path/to/imagenet_256 --output_dir /path/to/out --resume /path/to/out
```

A matched **baseline** (no alignment) is JiT's own `main_jit.py` with the same training args.
`--online_eval` logs FID/IS every `eval_freq` epochs to TensorBoard (tag `fid_cfg3.0_res256`).

### Self-REPA block indices by model size

The alignment reads an **early** online block (student, at the block where the in-context class tokens
have just entered, `≈⅓` depth) and a **deeper** EMA-teacher block (`≈⅔` depth). `λ`, `Δ`, the EMA decay,
and the effective batch do **not** change with model size — only `--model` and the two block indices do.

| model | depth | `--self_student_block` → `--self_teacher_block` | eval CFG\* |
|---|---|---|---|
| JiT-B/16 | 12 | **4 → 8**  (validated) | 3.0 |
| JiT-L/16 | 24 | **8 → 16** (validated) | 2.6 |
| JiT-H/16 | 32 | **10 → 20** (recommended) | 2.2 |

\* Evaluate each size at its optimal guidance; these are starting points (JiT's per-size CFG) — confirm
with a small CFG sweep at convergence. We have **run and reported B/16 and L/16**; the H/16 indices follow
the same `⅓→⅔`-depth rule (exact teacher index ≈20–21 is not critical) and are **untested**.

### Patch-32 variants / higher resolution

`JiT-{B,L,H}/32` are the **patch-size-32** variants — identical depth/width to their `/16` siblings, only
`patch_size` differs — intended for **512×512** training (`--img_size 512 --noise_scale 2.0`), where the
larger patch keeps the sequence length manageable (`512/32 = 16 → 256` tokens, same as `256/16`). Self-REPA
transfers unchanged: a `/32` model uses the **same student→teacher block indices as its `/16` sibling**
(B `4→8`, L `8→16`, H `10→20`), since depth is identical. We have not run the `/32` / 512-px setting.

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
