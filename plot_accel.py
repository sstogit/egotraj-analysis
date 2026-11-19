# -*- coding: utf-8 -*-
"""速度突变（accel）是否比速度本身更能解释误差。

定义：
    speed_hist = 历史 8 帧的平均速度（7 段位移的均值）
    speed_fut  = 未来 12 帧的平均速度（11 段位移之和 / 11）
    accel      = speed_fut - speed_hist     正 = 加速，负 = 减速

输出：
    1. |accel| 分档的 ADE 表
    2. 四组变量的方差解释比 η²（谁更能解释 ADE）
    3. 一张 2×2 组合图

用法（放在 EgoTraj-Bench 根目录）：
    py -3.9 plot_accel.py

只读 pkl，只写一张 PNG。
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
ROOT = Path(os.environ.get("EGOTRAJ_ROOT",
                         str(BASE.parents[1] if len(BASE.parents) > 1 else BASE)))

SAMPLE_DIR = BASE / "checkpoints" / "EgoTraj-TBD" / "samples"
if not SAMPLE_DIR.exists():
    SAMPLE_DIR = ROOT / "02-程序" / "EgoTraj-Bench" / "checkpoints" / "EgoTraj-TBD" / "samples"
SAMPLE_DIR = Path(os.environ.get("EGOTRAJ_SAMPLES", str(SAMPLE_DIR)))
OUT_DIR = ROOT / "03-结果" / "图片"


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


def eta2(y, g):
    """方差解释比：组间平方和 / 总平方和。越大说明这个分组越能解释 y。"""
    grand = y.mean()
    ss_tot = ((y - grand) ** 2).sum()
    ss_bet = 0.0
    for k in np.unique(g):
        yk = y[g == k]
        ss_bet += len(yk) * (yk.mean() - grand) ** 2
    return ss_bet / ss_tot if ss_tot > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description="速度突变 vs 速度：谁更能解释误差")
    ap.add_argument("--still", type=float, default=0.5)
    ap.add_argument("--bins", type=int, default=3, help="|accel| 分几档")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    d = load_pkls()
    fut_gt, fut_pred = d["fut_gt"], d["fut_pred"]
    hist_gt, hist_obs = d["hist_gt"], d["hist_obs"]

    dist = np.linalg.norm(fut_pred - fut_gt[:, None], axis=-1)
    ade = dist.mean(axis=-1).min(axis=1)
    noise = np.linalg.norm(hist_obs - hist_gt, axis=-1).mean(axis=1)

    travel = np.linalg.norm(np.diff(fut_gt, axis=1), axis=-1).sum(axis=1)
    n_fut = np.diff(fut_gt, axis=1).shape[1]          # 11 段
    n_hist = np.diff(hist_gt, axis=1).shape[1]        # 7 段

    speed_hist = np.linalg.norm(np.diff(hist_gt, axis=1), axis=-1).mean(axis=1)
    speed_fut = travel / n_fut

    accel = speed_fut - speed_hist
    abs_accel = np.abs(accel)

    keep = travel >= args.still
    print("-" * 66)
    print(f"minADE 均值 = {ade.mean():.4f}（应 0.2997）")
    print(f"静止样本 {np.sum(~keep)} 条 ({np.mean(~keep):.1%})，已剔除")
    ade, noise, speed_hist, speed_fut, accel, abs_accel, travel = (
        x[keep] for x in (ade, noise, speed_hist, speed_fut, accel, abs_accel, travel))
    n = len(ade)

    def qbin(x, k):
        edges = np.quantile(x, np.linspace(0, 1, k + 1)[1:-1])
        return np.clip(np.digitize(x, edges), 0, k - 1)

    K = args.bins
    g_speed = qbin(speed_hist, K)
    g_accel = qbin(abs_accel, K)
    g_travel = qbin(travel, 4)

    # 噪声分档（干净 + 四分位）
    is_clean = noise <= 1e-6
    rest = noise[~is_clean]
    edges_n = np.quantile(rest, [.25, .5, .75])
    g_noise = np.zeros(n, dtype=int)
    g_noise[~is_clean] = 1 + np.clip(np.digitize(rest, edges_n), 0, 3)

    # ---- |accel| 分档表 ----
    print("-" * 66)
    print(f"按 |accel| 分 {K} 档（正 = 加速，负 = 减速）")
    print(f"{'档':<6}{'n':>7}{'|accel|均值':>14}{'ADE':>10}{'占误差':>9}")
    total = ade.sum()
    for k in range(K):
        m = g_accel == k
        print(f"  {k:<5}{m.sum():>7d}{abs_accel[m].mean():>14.4f}"
              f"{ade[m].mean():>10.4f}{ade[m].sum() / total:>9.3f}")

    # ---- 方向：加速 vs 减速 ----
    print("-" * 66)
    print("按方向拆开（同一 |accel| 档内）")
    for k in range(K):
        for name, m in (("加速", (g_accel == k) & (accel > 0)),
                        ("减速", (g_accel == k) & (accel < 0))):
            if m.sum() >= 30:
                print(f"  档{k} {name}  n={m.sum():5d}  ADE={ade[m].mean():.4f}")

    # ---- η²：谁的区分力最强 ----
    print("-" * 66)
    print("方差解释比 η²（越大越能解释 ADE）：")
    scores = {
        "noise 5 档": eta2(ade, g_noise),
        f"speed 3 档": eta2(ade, g_speed),
        f"|accel| {K} 档": eta2(ade, g_accel),
        "travel 4 档": eta2(ade, g_travel),
    }
    for name, v in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<16}{v:.4f}")

    # ---- 画图 ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].hist(np.clip(accel, -0.6, 0.6), bins=80, color="tab:purple", alpha=.85)
    axes[0, 0].set_xlabel("accel = 未来速度 − 历史速度 (米/帧，截断到 ±0.6)")
    axes[0, 0].set_ylabel("样本数")
    axes[0, 0].set_title("速度突变分布")
    axes[0, 0].grid(alpha=.3)

    xs = np.arange(K)
    means = [ade[g_accel == k].mean() for k in range(K)]
    axes[0, 1].bar(xs, means, color="tab:red", alpha=.85)
    for k in range(K):
        axes[0, 1].text(k, means[k], f"{means[k]:.3f}\n(n={(g_accel == k).sum()})",
                        ha="center", va="bottom", fontsize=8)
    axes[0, 1].set_xticks(xs)
    axes[0, 1].set_xticklabels([f"档{k}" for k in range(K)])
    axes[0, 1].set_xlabel("|accel| 档位（越右变化越剧烈）")
    axes[0, 1].set_ylabel("minADE (m)")
    axes[0, 1].set_title("突变幅度 vs ADE")
    axes[0, 1].grid(axis="y", alpha=.3)

    names = list(scores.keys())
    vals = [scores[nm] for nm in names]
    order = np.argsort(vals)[::-1]
    axes[1, 0].barh([names[i] for i in order], [vals[i] for i in order],
                    color="tab:green", alpha=.85)
    axes[1, 0].set_xlabel("η²（方差解释比）")
    axes[1, 0].set_title("谁更能解释 ADE")
    axes[1, 0].grid(axis="x", alpha=.3)

    grid = np.full((K, K), np.nan)
    cnt = np.zeros((K, K), dtype=int)
    for a in range(K):
        for b in range(K):
            m = (g_speed == a) & (g_accel == b)
            cnt[a, b] = m.sum()
            if m.sum() >= 30:
                grid[a, b] = ade[m].mean()
    im = axes[1, 1].imshow(grid, cmap="magma", aspect="auto")
    axes[1, 1].set_xticks(range(K))
    axes[1, 1].set_xticklabels([f"突变{k}" for k in range(K)])
    axes[1, 1].set_yticks(range(K))
    axes[1, 1].set_yticklabels([f"速度{k}" for k in range(K)])
    axes[1, 1].set_xlabel("|accel| 档")
    axes[1, 1].set_ylabel("速度档")
    axes[1, 1].set_title("速度 × 突变 → ADE")
    vmax = np.nanmax(grid)
    for a in range(K):
        for b in range(K):
            if not np.isnan(grid[a, b]):
                axes[1, 1].text(b, a, f"{grid[a, b]:.3f}\n(n={cnt[a, b]})",
                                ha="center", va="center", fontsize=8,
                                color="w" if grid[a, b] < vmax * .7 else "k")
    plt.colorbar(im, ax=axes[1, 1], fraction=.046)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    fig_path = OUT_DIR / f"fig_accel{tag}.png"
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("-" * 66)
    print(f"已保存（仅此一个文件）: {fig_path}")


if __name__ == "__main__":
    main()
