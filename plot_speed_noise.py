# -*- coding: utf-8 -*-
"""速度 × 噪声 × 误差 的关系图（只读脚本）。

放在 EgoTraj-Bench 根目录运行：
    py -3.9 plot_speed_noise.py
    py -3.9 plot_speed_noise.py --clean 1e-3
    py -3.9 plot_speed_noise.py --tag run2

只读 pkl、只写一张 PNG，不生成/不覆盖任何 npz 或 csv。
"""

import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent
SAMPLE_DIR = BASE / "checkpoints" / "EgoTraj-TBD" / "samples"
OUT_DIR = BASE.parents[1] / "03-结果" / "图片"


def load_pkls():
    files = sorted(glob.glob(os.path.join(SAMPLE_DIR, "visual_mode_best_batch_*.pkl")))
    if not files:
        raise SystemExit(f"没有找到 pkl：{SAMPLE_DIR}")
    print(f"读取 {len(files)} 个 pkl ...")
    keys = ("fut_gt", "fut_pred", "hist_gt", "hist_obs")
    buf = {k: [] for k in keys}
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        for k in keys:
            buf[k].append(np.asarray(d[k], dtype=np.float32))
    return {k: np.concatenate(v, axis=0) for k, v in buf.items()}


def main():
    ap = argparse.ArgumentParser(description="速度×噪声×误差 关系图（只读）")
    ap.add_argument("--clean", type=float, default=1e-6, help="干净阈值，默认 1e-6")
    ap.add_argument("--still", type=float, default=0.5, help="行程 < 该值算静止，默认 0.5 m")
    ap.add_argument("--min-cell", type=int, default=30, help="热力图格内最少样本数")
    ap.add_argument("--tag", default="", help="输出文件名后缀，如 run2")
    args = ap.parse_args()

    d = load_pkls()
    fut_gt, fut_pred = d["fut_gt"], d["fut_pred"]
    hist_gt, hist_obs = d["hist_gt"], d["hist_obs"]
    n = len(fut_gt)

    # ---- 四个量 ----
    dist = np.linalg.norm(fut_pred - fut_gt[:, None], axis=-1)     # (N,K,12)
    ade = dist.mean(axis=-1).min(axis=1)                            # (N,)
    noise = np.linalg.norm(hist_obs - hist_gt, axis=-1).mean(axis=1)
    speed = np.linalg.norm(np.diff(hist_gt, axis=1), axis=-1).mean(axis=1)
    travel = np.linalg.norm(np.diff(fut_gt, axis=1), axis=-1).sum(axis=1)

    still = travel < args.still
    ade_norm = np.where(still, np.nan, ade / np.maximum(travel, 1e-6))

    print("-" * 60)
    print(f"N = {n}   minADE 均值 = {ade.mean():.4f}（应 0.2997）")
    print(f"静止样本 {still.sum()} 条 ({still.mean():.1%})")
    if abs(ade.mean() - 0.2997) > 5e-3:
        raise SystemExit("minADE 对不上，先别信后面的图")

    # ---- 分档 ----
    is_clean = noise <= args.clean
    rest = noise[~is_clean]
    if len(rest) < 40:
        raise SystemExit("噪声样本太少，无法分档")

    edges = np.quantile(rest, [0, .25, .5, .75, 1.0])
    nb = np.zeros(n, dtype=int)
    nb[~is_clean] = 1 + np.clip(np.digitize(rest, edges[1:-1]), 0, 3)

    sq = np.quantile(speed, [1 / 3, 2 / 3])
    sb = np.clip(np.digitize(speed, sq), 0, 2)

    NL = ["干净", "Q1", "Q2", "Q3", "Q4"]
    SL = ["慢", "中", "快"]

    # ---- 相关系数 ----
    def corr(x, y):
        m = ~np.isnan(y)
        return np.corrcoef(x[m], y[m])[0, 1]

    print("-" * 60)
    print("相关系数：")
    print(f"  speed vs noise      {corr(speed, noise):+.3f}")
    print(f"  speed vs ADE        {corr(speed, ade):+.3f}")
    print(f"  noise vs ADE        {corr(noise, ade):+.3f}")
    print(f"  speed vs ADE/行程   {corr(speed, ade_norm):+.3f}")
    print(f"  noise vs ADE/行程   {corr(noise, ade_norm):+.3f}")

    # ---- 3×5 网格（控制变量）----
    grid = np.full((3, 5), np.nan)
    cnt = np.zeros((3, 5), dtype=int)
    for i in range(3):
        for j in range(5):
            m = (sb == i) & (nb == j)
            cnt[i, j] = m.sum()
            if m.sum() >= args.min_cell:
                grid[i, j] = np.nanmean(ade_norm[m])

    print("-" * 60)
    print("平均 ADE/行程（行=速度档，列=噪声档；空=样本不足）")
    print("        " + "".join(f"{c:>10}" for c in NL))
    for i in range(3):
        cells = "".join(f"{('%.4f' % g) if not np.isnan(g) else '   -':>10}" for g in grid[i])
        print(f"{SL[i]:<8}{cells}")
    print("格内样本数：")
    for i in range(3):
        print(f"{SL[i]:<8}" + "".join(f"{c:>10}" for c in cnt[i]))

    # ---- 画图 ----
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    axes[0, 0].hist(speed, bins=60, color="tab:green", alpha=.85)
    axes[0, 0].set_xlabel("速度 (米/帧)")
    axes[0, 0].set_ylabel("样本数")
    axes[0, 0].set_title("速度分布")
    axes[0, 0].grid(alpha=.3)

    axes[0, 1].hist(np.clip(noise, 0, 1.5), bins=60, color="tab:gray", alpha=.85)
    axes[0, 1].set_xlabel("观测噪声 (m，截断到 1.5)")
    axes[0, 1].set_ylabel("样本数")
    axes[0, 1].set_title("噪声分布（双峰）")
    axes[0, 1].grid(alpha=.3)

    axes[0, 2].scatter(speed, np.clip(noise, 0, 2), s=2, alpha=.1, color="tab:gray")
    axes[0, 2].plot([speed[sb == i].mean() for i in range(3)],
                    [noise[sb == i].mean() for i in range(3)],
                    "o-", color="tab:red", lw=2, label="速度三分位均值")
    axes[0, 2].set_xlabel("速度 (米/帧)")
    axes[0, 2].set_ylabel("噪声 (m)")
    axes[0, 2].set_title(f"速度 vs 噪声（r={corr(speed, noise):+.2f}）")
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=.3)

    xs = np.arange(3)
    axes[1, 0].bar(xs - 0.2, [ade[sb == i].mean() for i in range(3)],
                   width=.4, color="tab:blue")
    axes[1, 0].set_ylabel("minADE (m)", color="tab:blue")
    ax2 = axes[1, 0].twinx()
    ax2.bar(xs + 0.2, [np.nanmean(ade_norm[sb == i]) for i in range(3)],
            width=.4, color="tab:orange", alpha=.85)
    ax2.set_ylabel("ADE / 行程", color="tab:orange")
    axes[1, 0].set_xticks(xs)
    axes[1, 0].set_xticklabels(SL)
    axes[1, 0].set_title("速度档 vs 误差（绝对 / 相对）")
    axes[1, 0].grid(axis="y", alpha=.3)

    im = axes[1, 1].imshow(grid, cmap="viridis", aspect="auto")
    axes[1, 1].set_xticks(range(5))
    axes[1, 1].set_xticklabels(NL, rotation=15)
    axes[1, 1].set_yticks(range(3))
    axes[1, 1].set_yticklabels(SL)
    axes[1, 1].set_xlabel("噪声档位")
    axes[1, 1].set_ylabel("速度档位")
    axes[1, 1].set_title("平均 ADE/行程（控制变量）")
    vmax = np.nanmax(grid)
    for i in range(3):
        for j in range(5):
            if not np.isnan(grid[i, j]):
                axes[1, 1].text(j, i, f"{grid[i, j]:.3f}\n(n={cnt[i, j]})",
                                ha="center", va="center", fontsize=8,
                                color="w" if grid[i, j] < vmax * 0.7 else "k")
    plt.colorbar(im, ax=axes[1, 1], fraction=.046)

    for i in range(3):
        ys = [np.nanmean(ade_norm[(sb == i) & (nb == j)]) for j in range(5)]
        axes[1, 2].plot(range(5), ys, "o-", lw=2, label=f"{SL[i]}速")
    axes[1, 2].set_xticks(range(5))
    axes[1, 2].set_xticklabels(NL, rotation=15)
    axes[1, 2].set_xlabel("噪声档位")
    axes[1, 2].set_ylabel("ADE / 行程")
    axes[1, 2].set_title("控制速度后，噪声的净效应")
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=.3)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    fig_path = OUT_DIR / f"fig_speed_noise{tag}.png"
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("-" * 60)
    print(f"已保存（仅此一个文件）: {fig_path}")


if __name__ == "__main__":
    main()
