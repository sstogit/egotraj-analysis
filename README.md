# egotraj-analysis

Companion scripts for reproducing and stratifying **EgoTraj-Bench**.

- Upstream code: `github.com/zoeyliu1999/EgoTraj-Bench` @ `3c519c8` — the scripts here do not modify it.
- Data: HuggingFace `ZoeyLIU1999/EgoTraj-Bench` -> `L2-processed/EgoTraj-TBD-v1.1`
- Paper: arXiv:2510.00405

## What this is

Upstream code reports a single average minADE/minFDE. These scripts ask a different
question: **where does the error actually come from?**

Results on the EgoTraj-TBD test split (9,828 samples, K=20):

| stratum | n | share | minADE | share of total error |
|---|---:|---:|---:|---:|
| clean (noise = 0) | 6,908 | 70.3% | 0.1920 | 45.0% |
| Q1 | 730 | 7.4% | 0.3695 | 9.2% |
| Q2 | 730 | 7.4% | 0.4778 | 11.8% |
| Q3 | 730 | 7.4% | 0.6014 | 14.9% |
| Q4 | 730 | 7.4% | 0.7696 | 19.1% |

Variance explained (eta-squared) by candidate variables:
observation noise **0.39**, travel length 0.06, speed change 0.06, history speed 0.02.

Occlusion is a step, not a gradient: samples missing a single frame already show
**2.92x** the error of clean ones (0.5883 m vs 0.2016 m).

## Scripts

| file | purpose |
|---|---|
| `analyze_per_sample.py` | stratify test samples by observation noise; writes `per_sample_*.npz/.csv`, figures, and `summary.csv` |
| `occ_eval.py` | occlusion-stratified evaluation for any model's `samples/*.pkl` |
| `plot_speed_noise.py` | speed x noise grid |
| `plot_travel_strata.py` | travel-length-stratified analysis |
| `plot_accel.py` | acceleration analysis and eta-squared ranking |
| `plot_occlusion_duration.py` | occlusion duration vs ADE |
| `inspect_data.py` | dataset health check |

## Setup

```bash
pip install numpy matplotlib
# torch is only needed for the upstream evaluation that produces samples/*.pkl
```

## Paths: set `EGOTRAJ_ROOT`

The scripts read the test split and write results relative to a project root:

```
<EGOTRAJ_ROOT>/
├── 01-数据/L2-processed/EgoTraj-TBD-v1.1/egotraj_tbd_test.npz
├── 02-程序/EgoTraj-Bench/          # upstream clone (checkpoints/, data/, samples/)
└── 03-结果/                        # outputs are written here
```

Point the environment variable at that root:

```powershell
$env:EGOTRAJ_ROOT = "C:\path\to\EgoTraj-Project"
```

```bash
export EGOTRAJ_ROOT=/path/to/EgoTraj-Project
```

If `EGOTRAJ_ROOT` is unset, each script falls back to its own directory's parent,
which is what happens when the scripts sit inside `02-程序/EgoTraj-Bench/`.

The per-sample predictions live in the upstream clone at
`02-程序/EgoTraj-Bench/checkpoints/EgoTraj-TBD/samples/`. Scripts look there by
default; point `EGOTRAJ_SAMPLES` elsewhere if your layout differs:

```powershell
$env:EGOTRAJ_SAMPLES = "D:\runs\my_model\samples"
```

## Usage

```bash
# 1) run the upstream evaluation once, to produce per-sample predictions
cd "$EGOTRAJ_ROOT/02-程序/EgoTraj-Bench"
python scripts/eval_biflow.py --ckpt_dir checkpoints/EgoTraj-TBD --fold_name tbd \
    --data_source original --data_dir ./data/egotraj --gpu 0 --mode best

# 2) stratify by observation noise
python analyze_per_sample.py --fde official --clean 1e-6

# 3) occlusion-stratified evaluation (validates the expected minADE first)
python occ_eval.py --tag biflow --expect 0.2997
```

`occ_eval.py` takes `--samples <dir>` so that any model whose predictions are dumped
in the same `visual_mode_best_batch_*.pkl` format can be evaluated with one command.

## Notes

- `load_predictions()` sorts batch files by **numeric** index. Lexicographic sorting
  puts `batch_10` before `batch_2`, which silently misaligns predictions against the
  L2 arrays.
- `analyze_per_sample.py` reads the export with `rotate=True` applied by the upstream
  dataloader, so its coordinates are per-sample rotated; the `.npz` test split is not.
  Rotational-invariant quantities (path length) match to 4e-6.
- Reproduced numbers: minADE 0.2997 / minFDE 0.4403 at K=20, against the README's
  0.30 / 0.44. The 0.19 / 0.27 in the paper body belongs to an earlier data release.
