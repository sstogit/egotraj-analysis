# -*- coding: utf-8 -*-
"""把「速度效应」和「尺度效应」分开：按行程分层后再看速度。

动机：
    ade_norm = ade / travel 会让小行程的样本比值虚高，
    所以"慢=相对误差大"可能只是分母小，不是模型更差。

做法：
    把 travel 切成 4 档（四分位），在同一档内比较速度档的 ADE。
    固定了 travel，绝对 ADE 的差异就不再受尺度影响。

用法（放在 EgoTraj-Bench 根目录）：
    py -3.9 plot_travel_strata.py
    py -3.9 plot_travel_strata.py --still 0.5 --min-cell 30

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
    ap = argparse.ArgumentParser(description="按行程分层，分离速度效应与尺度效应")
    ap.add_argument("--still", type=float, default=0.5, help="行程 < 该值算静止")
    ap.add_argument("--min-cell", type=int, default=30, help="单元格最少样本数")
    ap.add_argument("--tag", default="", help="输出文件名后缀")
    args = ap.parse_args()

    d = load_pkls()
    fut_gt, fut_pred = d["fut_gt"], d["fut_pred"]
    hist_gt, hist_obs = d["hist_gt"], d["hist_obs"]

    dist = np.linalg.norm(fut_pred - fut_gt[:, None], axis=-1)
    ade = dist.mean(axis=-1).min(axis=1)
    noise = np.linalg.norm(hist_obs - hist_gt, axis=-1).mean(axis=1)
    speed = np.linalg.norm(np.diff(hist_gt, axis=1), axis=-1).mean(axis=1)
    travel = np.linalg.norm(np.diff(fut_gt, axis=1), axis=-1).sum(axis=1)

    keep = travel >= args.still          # 只分析非静止样本
    print("-" * 66)
    print(f"minADE 均值 = {ade.mean():.4f}（应 0.2997）")
    print(f"静止样本 {np.sum(~keep)} 条 ({np.mean(~keep):.1%})，已剔除")

    ade, noise, speed, travel = ade[keep], noise[keep], speed[keep], travel[keep]
    n = len(ade)

    # ---- 分档 ----
    sq = np.quantile(speed, [1 / 3, 2 / 3])
    sb = np.clip(np.digitize(speed, sq), 0, 2)          # 速度 3 档

    tq = np.quantile(travel, [.25, .5, .75])
    tb = np.digitize(travel, tq)                        # 行程 4 档
    SL = ["慢", "中", "快"]
    TL = ["T1 短", "T2", "T3", "T4 长"]

    def corr(x, y):
        return np.corrcoef(x, y)[0, 1]

    print("-" * 66)
    print(f"speed vs travel  r = {corr(speed, travel):+.3f}   "
          f"（高 → 分层后速度差异会被 travel 吸收）")
    print(f"speed vs ADE     r = {corr(speed, ade):+.3f}")
    print(f"travel vs ADE    r = {corr(travel, ade):+.3f}")

    # ---- 控制前：只看速度档 ----
    raw = [ade[sb == s].mean() for s in range(3)]
    print("-" * 66)
    print("未控制 travel：按速度档直接看 ADE")
    for s in range(3):
        print(f"  {SL[s]}  n={np.sum(sb == s):5d}   ADE={raw[s]:.4f}")

    # ---- 控制后：行程档内比较速度 ----
    print("-" * 66)
    print("控制 travel：同一行程档内比较速度档的 ADE")
    print(f"{'行程档':<8}{'n':>7}" + "".join(f"{s + ' ADE':>10}" for s in SL))
    cell = np.full((4, 3), np.nan)
    cnt = np.zeros((4, 3), dtype=int)
    for t in range(4):
        m_t = tb == t
        line = f"{TL[t]:<8}{np.sum(m_t):>7d}"
        for s in range(3):
            m = m_t & (sb == s)
            cnt[t, s] = m.sum()
            if m.sum() >= args.min_cell:
                cell[t, s] = ade[m].mean()
            line += f"{('%.4f' % cell[t, s]) if not np.isnan(cell[t, s]) else '-':>10}"
        print(line)

    spreads = [np.nanmax(cell[t]) - np.nanmin(cell[t])
               for t in range(4) if np.sum(~np.isnan(cell[t])) >= 2]
    spread_raw = max(raw) - min(raw)
    spread_ctrl = float(np.mean(spreads)) if spreads else float("nan")

    print("-" * 66)
    print(f"速度档之间的 ADE 差：未控制 {spread_raw:.4f} m  →  控制后 {spread_ctrl:.4f} m")
    if spread_ctrl < spread_raw * 0.5:
        print("  结论：差距大幅缩小 → 之前的「速度效应」主要来自尺度")
    else:
        print("  结论：差距仍在 → 速度有独立效应")

    # ---- 画图 ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].scatter(travel, speed, s=2, alpha=.1, color="tab:gray")
    for q in tq:
        axes[0, 0].axvline(q, color="tab:red", ls="--", lw=1)
    axes[0, 0].set_xlim(0, np.quantile(travel, .98))
    axes[0, 0].set_xlabel("行程 travel (m)")
    axes[0, 0].set_ylabel("速度 (米/帧)")
    axes[0, 0].set_title(f"速度 vs 行程（r={corr(speed, travel):+.2f}）")
    axes[0, 0].grid(alpha=.3)

    xs = np.arange(4)
    for s in range(3):
        axes[0, 1].plot(xs, cell[:, s], "o-", lw=2, label=f"{SL[s]}速")
    axes[0, 1].set_xticks(xs)
    axes[0, 1].set_xticklabels(TL, rotation=10)
    axes[0, 1].set_xlabel("行程档位")
    axes[0, 1].set_ylabel("minADE (m)")
    axes[0, 1].set_title("控制行程后：速度的净效应")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=.3)

    im = axes[1, 0].imshow(cell, cmap="viridis", aspect="auto")
    axes[1, 0].set_xticks(range(3))
    axes[1, 0].set_xticklabels(SL)
    axes[1, 0].set_yticks(range(4))
    axes[1, 0].set_yticklabels(TL)
    axes[1, 0].set_xlabel("速度档位")
    axes[1, 0].set_ylabel("行程档位")
    axes[1, 0].set_title("minADE 网格（n<阈值 留空）")
    vmax = np.nanmax(cell)
    for t in range(4):
        for s in range(3):
            if not np.isnan(cell[t, s]):
                axes[1, 0].text(s, t, f"{cell[t, s]:.3f}\n(n={cnt[t, s]})",
                                ha="center", va="center", fontsize=8,
                                color="w" if cell[t, s] < vmax * .7 else "k")
    plt.colorbar(im, ax=axes[1, 0], fraction=.046)

    axes[1, 1].bar([0, 1], [spread_raw, spread_ctrl], color=["tab:gray", "tab:green"])
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_xticklabels(["未控制行程", "控制行程后"])
    axes[1, 1].set_ylabel("速度档之间的 ADE 差 (m)")
    axes[1, 1].set_title("尺度效应占了多少")
    axes[1, 1].grid(axis="y", alpha=.3)
    for i, v in enumerate([spread_raw, spread_ctrl]):
        axes[1, 1].text(i, v, f"{v:.4f}", ha="center", va="bottom")

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    fig_path = OUT_DIR / f"fig_travel_strata{tag}.png"
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("-" * 66)
    print(f"已保存（仅此一个文件）: {fig_path}")


if __name__ == "__main__":
    main()
