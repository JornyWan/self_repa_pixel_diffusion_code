# Self-REPA Scaling Experiments — Executable Plan

A step-by-step plan to train a matched **JiT baseline + Self-REPA** pair at every remaining model size
and report FID/IS at convergence. Designed to be run by an automation agent, **one experiment at a time**,
with **no code changes** — every experiment differs only in command-line flags.

## Status

| # | size | res | status |
|---|------|-----|--------|
| 0 | B/16 | 256 | ✅ **DONE** — Self-REPA beats JiT (e.g. ep100: JiT FID 7.38 → Self-REPA 6.94) |
| 1 | L/16 | 256 | ⬜ to run |
| 2 | H/16 | 256 | ⬜ to run |
| 3 | B/32 | 512 | ⬜ to run |
| 4 | L/32 | 512 | ⬜ to run |
| 5 | H/32 | 512 | ⬜ to run |

`/16` = patch-16 (256×256). `/32` = patch-32, used for **512×512** (same token count as 256/16, so
per-step cost is similar). Each experiment = **two training runs** (baseline + Self-REPA) + a final eval.

## Rules for the executing agent

1. **Run experiments top-to-bottom, one at a time.** For each: (A) train baseline → (B) train Self-REPA
   → (C) eval both at the per-size CFG → (D) fill the results row → check the box. Independent experiments
   *may* run in parallel if spare GPUs exist, but never run the two arms of one size on different
   effective batch / GPU counts.
2. **No code edits.** Only the flags in the tables below change. Baseline uses `main_jit.py`; Self-REPA
   uses `jit_selfrepa_train.py`. Everything else is identical between the two arms of a size.
3. **Do not change the recipe mid-run** (λ, Δ, EMA, blr, batch, dropout are fixed). If Self-REPA looks
   tied early, that is expected — the gain is small and clearest at convergence; **do not** enable spatial
   normalization or raise λ (both were *worse* in the original study).
4. **Resume on any interruption**: every run takes `--resume <output_dir>` and checkpoints every 5 epochs,
   so re-launching the exact same command continues from the last checkpoint. Never delete `checkpoint-*.pth`.
5. **Matched comparison**: baseline and Self-REPA of a given size must use the **same** effective batch,
   LR, schedule, data, and hardware — only the alignment objective differs.
6. **Success criterion (report regardless):** at 600 epochs and the size's optimal CFG, `Self-REPA FID ≤
   baseline FID`. Also log IS. If Self-REPA is *not* better at convergence, do **not** tweak the recipe —
   record the numbers and flag it for the authors.

## Prerequisites (once)

```bash
conda env create -f environment.yaml && conda activate jit   # torch 2.5.1 / cu124
```
- **Data**: 256-px runs use ImageNet at `--img_size 256`; 512-px runs use the **same ImageNet** at
  `--img_size 512` (center-cropped). If your copy is pre-resized to 256, point the 512 runs at the
  full-resolution ImageNet.
- FID reference stats are bundled (`fid_stats/jit_in{256,512}_stats.npz`) and selected automatically by
  `--img_size` — nothing to configure.
- Quick sanity check: `bash smoke_selfrepa.sh`.

## Shared settings (identical for every run)

```
effective batch = 1024   |   blr = 5e-5   |   epochs = 600   |   warmup_epochs = 5
AdamW            |   label_drop_prob = 0.1   |   online eval every 40 ep (monitoring only)
Self-REPA extra:  --self_lambda 0.1  --self_teacher_dt 0.5  --self_teacher_ema 0.9999
```

Set per-GPU `--batch_size` (BS) and `--accum_iter` (ACC) so that **BS × NGPU × ACC = 1024**.
Suggested BS on 80-GB GPUs: B,L,/32 → 32 (ACC=`1024/(32·NGPU)`); H/16, H/32 → 16 (reduce if OOM, raise ACC).

## Per-size parameters

| size | `--model` | `--img_size` | `--noise_scale` | `--self_student_block` → `--self_teacher_block` | `--gen_bsz` | final CFG |
|------|-----------|--------------|-----------------|--------------------------------------------------|-------------|-----------|
| L/16 | `JiT-L/16` | 256 | 1.0 | **8 → 16** | 64 | sweep around **2.6** |
| H/16 | `JiT-H/16` | 256 | 1.0 | **10 → 20** | 64 | sweep around **2.2** |
| B/32 | `JiT-B/32` | 512 | 2.0 | **4 → 8**  | 64 | sweep **2.0–3.2** |
| L/32 | `JiT-L/32` | 512 | 2.0 | **8 → 16** | 64 | sweep **2.0–3.2** |
| H/32 | `JiT-H/32` | 512 | 2.0 | **10 → 20** | 64 | sweep **2.0–3.2** |

> Block rule (already validated at B/16=4→8, L/16=8→16): **student = `in_context_start` (≈⅓ depth, where
> the class tokens enter), teacher ≈⅔ depth.** The `/32` variant shares its `/16` sibling's depth, so it
> reuses the same block indices. The CFG "center" is a starting point from JiT's per-size guidance; the
> reported number comes from the sweep in Step C.

## Set these once

```bash
export NGPU=32                                  # your GPU count (use the SAME for both arms of a size)
export DATA256=/path/to/imagenet               # ImageNet, cropped to 256 at load
export DATA512=/path/to/imagenet               # same ImageNet, cropped to 512 (full-res source)
export OUT=/path/to/checkpoints                # output root
export BS=32                                    # per-GPU batch (16 for H/*); ACC below keeps eff-batch 1024
```

---

## Experiment template

For a size with model `M`, resolution `R`, noise `N`, blocks `s→t`, per-GPU batch `BS`:
`ACC = 1024 / (BS × NGPU)` and `DATA = $DATA256` (R=256) or `$DATA512` (R=512).

**(A) Baseline** → `main_jit.py`, output dir `$OUT/<tag>_base`:
```bash
ACC=$((1024/(BS*NGPU)))
torchrun --nproc_per_node=$NGPU --standalone main_jit.py \
    --model M --img_size R --noise_scale N \
    --batch_size $BS --accum_iter $ACC --blr 5e-5 --epochs 600 --warmup_epochs 5 \
    --online_eval --eval_freq 40 --save_last_freq 5 \
    --gen_bsz 64 --num_images 50000 --cfg <center> --interval_min 0.1 --interval_max 1.0 \
    --data_path $DATA --output_dir $OUT/<tag>_base --resume $OUT/<tag>_base
```

**(B) Self-REPA** → `jit_selfrepa_train.py`, output dir `$OUT/<tag>_selfrepa` (add the 5 `--self_*` flags):
```bash
ACC=$((1024/(BS*NGPU)))
torchrun --nproc_per_node=$NGPU --standalone jit_selfrepa_train.py \
    --model M --img_size R --noise_scale N \
    --self_student_block s --self_teacher_block t \
    --self_lambda 0.1 --self_teacher_dt 0.5 --self_teacher_ema 0.9999 \
    --batch_size $BS --accum_iter $ACC --blr 5e-5 --epochs 600 --warmup_epochs 5 \
    --online_eval --eval_freq 40 --save_last_freq 5 \
    --gen_bsz 64 --num_images 50000 --cfg <center> --interval_min 0.1 --interval_max 1.0 \
    --data_path $DATA --output_dir $OUT/<tag>_selfrepa --resume $OUT/<tag>_selfrepa
```

**(C) Final eval — per-arm CFG sweep.** After 600 ep, for **each** arm run `--evaluate_gen` over a small
CFG grid (8k images to pick, then 50k at the best), on the same hardware. Baseline via `main_jit.py`,
Self-REPA via `jit_selfrepa_train.py` **with the same `--self_*` flags** (needed to load its checkpoint):
```bash
for CFG in <grid>; do
  torchrun --nproc_per_node=2 --standalone <main_jit.py | jit_selfrepa_train.py [+ --self_* flags]> \
    --model M --img_size R --noise_scale N --evaluate_gen --seed 0 \
    --gen_bsz 64 --num_images 8000 --cfg $CFG --interval_min 0.1 --interval_max 1.0 \
    --data_path $DATA --resume $OUT/<tag>_<arm> --output_dir $OUT/<tag>_<arm>
done
# pick the CFG with the lowest 8k-FID, then re-run that CFG once at --num_images 50000 for the reported value.
```
*(If you already have CFG/EMA-sweep tooling, use it — the centers/grids above are the only inputs.)*

**(D) Record** the 50k FID/IS of both arms in the results table and check the box.

---

## Experiments (ordered)

### 1 — L/16 (256px)   tag `l16`
`M=JiT-L/16  R=256  N=1.0  s=8 t=16  BS=32  DATA=$DATA256  CFG grid: 2.2 2.4 2.6 2.8`
- [ ] (A) baseline trained to 600 ep  → `$OUT/l16_base`
- [ ] (B) Self-REPA trained to 600 ep → `$OUT/l16_selfrepa`
- [ ] (C) CFG sweep both arms, 50k at best
- [ ] (D) results recorded

### 2 — H/16 (256px)   tag `h16`
`M=JiT-H/16  R=256  N=1.0  s=10 t=20  BS=16  DATA=$DATA256  CFG grid: 1.8 2.0 2.2 2.4`
- [ ] (A) baseline → `$OUT/h16_base`
- [ ] (B) Self-REPA → `$OUT/h16_selfrepa`
- [ ] (C) CFG sweep + 50k
- [ ] (D) recorded

### 3 — B/32 (512px)   tag `b32`
`M=JiT-B/32  R=512  N=2.0  s=4 t=8  BS=32  DATA=$DATA512  CFG grid: 2.0 2.4 2.8 3.2`
- [ ] (A) baseline → `$OUT/b32_base`
- [ ] (B) Self-REPA → `$OUT/b32_selfrepa`
- [ ] (C) CFG sweep + 50k
- [ ] (D) recorded

### 4 — L/32 (512px)   tag `l32`
`M=JiT-L/32  R=512  N=2.0  s=8 t=16  BS=32  DATA=$DATA512  CFG grid: 2.0 2.4 2.8 3.2`
- [ ] (A) baseline → `$OUT/l32_base`
- [ ] (B) Self-REPA → `$OUT/l32_selfrepa`
- [ ] (C) CFG sweep + 50k
- [ ] (D) recorded

### 5 — H/32 (512px)   tag `h32`
`M=JiT-H/32  R=512  N=2.0  s=10 t=20  BS=16  DATA=$DATA512  CFG grid: 1.8 2.2 2.6 3.0`
- [ ] (A) baseline → `$OUT/h32_base`
- [ ] (B) Self-REPA → `$OUT/h32_selfrepa`
- [ ] (C) CFG sweep + 50k
- [ ] (D) recorded

---

## Results (fill in — 50k FID/IS at each size's best CFG)

| size | res | JiT FID | JiT IS | Self-REPA FID | Self-REPA IS | ΔFID (base−self) | best CFG |
|------|-----|---------|--------|---------------|--------------|------------------|----------|
| B/16 | 256 | *(done — fill final)* | | | | | 3.0 |
| L/16 | 256 | | | | | | |
| H/16 | 256 | | | | | | |
| B/32 | 512 | | | | | | |
| L/32 | 512 | | | | | | |
| H/32 | 512 | | | | | | |

**Monitoring during a run:** online-eval FID/IS are logged to TensorBoard every 40 ep (tags
`fid_cfg<c>_res<r>` / `is_...`). Use them to confirm both arms descend normally and to watch the
Self-REPA−baseline gap; the **verdict is the Step-C converged + best-CFG number**, not an early online point.
