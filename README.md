# egotraj-analysis

Companion scripts for reproducing and stratifying **EgoTraj-Bench**.

- Upstream code: `github.com/zoeyliu1999/EgoTraj-Bench` @ `3c519c8` — the scripts here do not modify it.
- Data: HuggingFace `ZoeyLIU1999/EgoTraj-Bench` -> `L2-processed/EgoTraj-TBD-v1.1`
- Paper: arXiv:2510.00405

## What this is

Upstream code reports a single average minADE/minFDE. These scripts ask a different
question: **where does the error actually come from?**

## A data caveat found the hard way

The 9,828 test samples are **not 9,828 pedestrians**:

| subset | n | share | minADE | minFDE | share of error mass |
|---|---:|---:|---:|---:|---:|
| ego row (robot itself) | 6,908 | 70.3% | 0.1920 | 0.3222 | 45.0% |
| pedestrians | 2,920 | 29.7% | 0.5546 | 0.7198 | 55.0% |
| all | 9,828 | 100% | 0.2997 | 0.4403 | 100% |

The 6,908 zero-noise rows are the **ego row itself**: exactly one per scene, all eight
history frames valid, and `img_x` is `640` for every frame (the image centre of a
1280x720 frame) — where the camera projects onto itself. Every other row has
`img_x != 640` in all frames. This matches the paper's setting, where the robot
trajectory comes from odometry and is always observable.

**Consequence:** stratifying naively on observation noise splits "robot vs pedestrian",
not "clean vs noisy". That produces a spurious eta-squared of 0.39. Restricted to the
2,920 pedestrian samples the numbers are:

| stratum | n | noise (m) | minADE | minFDE |
|---|---:|---:|---:|---:|
| Q1 | 730 | 0.3654 | 0.3695 | 0.5130 |
| Q2 | 730 | 0.6180 | 0.4778 | 0.6397 |
| Q3 | 730 | 0.9588 | 0.6014 | 0.7871 |
| Q4 | 730 | 1.5994 | 0.7696 | 0.9391 |

Variance explained (eta-squared), pedestrians only:

| variable | eta-squared |
|---|---:|
| travel length | 0.196 |
| history speed | 0.198 |
| |accel| | 0.119 |
| **observation noise** | **0.080** |
| missing-frame count | 0.038 |

Missing observations within pedestrians: 0.3569 m (n=426, fully observed) vs 0.5883 m
(n=2,494, any missing) — **1.65x**, not a step.

Note that ADE is in metres, so longer motion mechanically inflates absolute error; the
travel/speed entries are a scale effect, not a causal driver.

## Scripts

| file | purpose |
|---|---|
| `analyze_per_sample.py` | stratify test samples by observation noise; writes `per_sample_*.npz/.csv`, figures, and `summary.csv` |
| `occ_eval.py` | occlusion-stratified evaluation for any model's per-sample predictions |
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

`occ_eval.py` takes `--samples <dir>` so that any model whose per-sample predictions are
dumped in the benchmark's export format can be evaluated with one command.

## Notes

- `load_predictions()` sorts batch files by **numeric** index. Lexicographic sorting
  puts `batch_10` before `batch_2`, which silently misaligns predictions against the
  L2 arrays.
- The exported `hist_gt` has the upstream dataloader's `rotate=True` applied, so its
  coordinates are per-sample rotated; the `.npz` test split is not. Rotation-invariant
  quantities (path length) match to 4e-6.
- Reproduced numbers: minADE 0.2997 / minFDE 0.4403 at K=20, against the README's
  0.30 / 0.44. The 0.19 / 0.27 in the paper body belongs to an earlier data release.
