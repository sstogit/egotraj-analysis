# egotraj-analysis

Companion scripts for reproducing and stratifying **EgoTraj-Bench**.

- Upstream: github.com/zoeyliu1999/EgoTraj-Bench @ 3c519c8 — these scripts do not modify it.
- Data: HuggingFace ZoeyLIU1999/EgoTraj-Bench -> L2-processed/EgoTraj-TBD-v1.1

## Scripts

| file | purpose |
|---|---|
| analyze_per_sample.py | stratify test samples by observation noise; writes summary.csv |
| occ_eval.py | occlusion-stratified evaluation for any model's samples/*.pkl |
| plot_speed_noise.py | speed x noise grid |
| plot_travel_strata.py | travel-length-stratified analysis |
| plot_accel.py | acceleration analysis and eta-squared ranking |
| plot_occlusion_duration.py | occlusion duration vs ADE |
| inspect_data.py | dataset health check |
