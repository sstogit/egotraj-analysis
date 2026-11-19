# -*- coding: utf-8 -*-
"""遮挡「时长」与 ADE 的关系。

三个"时长"定义：
    n_missing    : 8 帧里一共丢了几帧（0~8）
    max_run      : 最长连续丢失帧数
    tail_missing : 末尾连续丢失帧数（从最后一帧往前数）

最后一个是关键：预测紧接在最后一帧之后，所以"最近是否丢了"可能比"一共丢了几帧"更要命。

数据来源：
    预测与真值  ← checkpoints\\...\\samples\\*.pkl（按 batch 编号数值排序，与 npz 对齐）
    可见性掩码  ← 01-数据\\L2-processed\\EgoTraj-TBD-v1.1\\egotraj_tbd_test.npz

用法（放在 EgoTraj-Bench 根目录）：
    py -3.9 plot_occlusion_duration.py

只读，只写一张 PNG。
"""

import argparse
import glob
import os
import pickle
import re
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
NPZ = ROOT / "01-数据" / "L2-processed" / "EgoTraj-TBD-v1.1" / "egotraj_tbd_test.npz"
OUT_DIR = ROOT / "03-结果" / "图片"


def load_pkls():
    files = glob.glob(os.path.join(SAMPLE_DIR, "visual_mode_best_batch_*.pkl"))
    # 关键：按编号数值排序，否则 batch_10 会排在 batch_2 前面，与 npz 错位
    files.sort(key=lambda p: int(re.search(r"batch_(\d+)", p).group(1)))
    print(f"读取 {len(files)} 个 pkl（按编号数值排序）")
    keys = ("fut_gt", "fut_pred")
    buf = {k: [] for k in keys}
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        for k in keys:
            buf[k].append(np.asarray(d[k], dtype=np.float32))
    return {k: np.concatenate(v, axis=0) for k, v in buf.items()}


def longest_run_zero(row):
    best = cur = 0
    for v in row:
        cur = cur + 1 if v == 0 else 0
        best = max(best, cur)
    return best


def tail_zero(row):
    k = 0
    for v in row[::-1]:
        if v == 0:
            k += 1
        else:
            break
    return k


def eta2(y, g):
    grand = y.mean()
    ss_tot = ((y - grand) ** 2).sum()
    ss_bet = sum(len(y[g == k]) * (y[g == k].mean() - grand) ** 2 for k in np.unique(g))
    return ss_bet / ss_tot if ss_tot > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description="遮挡时长 vs ADE")
    ap.add_argument("--min-cell", type=int, default=30)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    d = load_pkls()
    fut_gt, fut_pred = d["fut_gt"], d["fut_pred"]

    z = np.load(NPZ)
    mask = z["all_obs"][:, :, 5]          # (N,8) 1=可见
    if len(mask) != len(fut_gt):
        raise SystemExit(f"数量不一致：mask {len(mask)} vs pred {len(fut_gt)}")

    dist = np.linalg.norm(fut_pred - fut_gt[:, None], axis=-1)
    ade = dist.mean(axis=-1).min(axis=1)
    travel = np.linalg.norm(np.diff(fut_gt, axis=1), axis=-1).sum(axis=1)
    still = travel < 0.5
    ade_norm = np.where(still, np.nan, ade / np.maximum(travel, 1e-6))

    n_missing = (mask == 0).sum(axis=1)
    max_run = np.array([longest_run_zero(r) for r in mask])
    tail_missing = np.array([tail_zero(r) for r in mask])

    print("-" * 66)
    print(f"minADE 均值 = {ade.mean():.4f}（应 0.2997）")
    print(f"完全无缺失 {(n_missing == 0).sum()} 条 ({(n_missing == 0).mean():.1%})")

    # ---- 三张表 ----
    def table(name, g, kmax):
        print("-" * 66)
        print(f"{name}")
        print(f"{'值':<6}{'n':>7}{'占比':>8}{'ADE':>9}{'ADE/行程':>11}")
        for k in range(kmax + 1):
            m = g == k
            if m.sum() == 0:
                continue
            print(f"{k:<6}{m.sum():>7d}{m.mean():>8.3f}{ade[m].mean():>9.4f}"
                  f"{np.nanmean(ade_norm[m]):>11.4f}")

    table("按缺失总帧数 n_missing", n_missing, 8)
    table("按最长连续缺失 max_run", max_run, 8)
    table("按末尾连续缺失 tail_missing", tail_missing, 8)

    # ---- η² ----
    print("-" * 66)
    print("方差解释比 η²（越高越能解释 ADE）：")
    scores = {
        "n_missing": eta2(ade, n_missing),
        "max_run": eta2(ade, max_run),
        "tail_missing": eta2(ade, tail_missing),
    }
    for k, v in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<14}{v:.4f}")

    # ---- 二维：有没有缺失 × 末尾是否缺失 ----
    print("-" * 66)
    print("交叉：总缺失帧数 × 末尾是否缺失")
    any_missing = (n_missing > 0).astype(int)
    tail_any = (tail_missing > 0).astype(int)
    for a in (0, 1):
        for b in (0, 1):
            m = (any_missing == a) & (tail_any == b)
            if m.sum() >= args.min_cell:
                print(f"  有缺失={a} 末尾丢={b}  n={m.sum():5d}  ADE={ade[m].mean():.4f}")

    # ---- 画图 ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for ax, g, name, kmax in ((axes[0, 0], n_missing, "缺失总帧数", 8),
                              (axes[0, 1], max_run, "最长连续缺失", 8),
                              (axes[1, 0], tail_missing, "末尾连续缺失", 8)):
        ks = [k for k in range(kmax + 1) if (g == k).sum() >= args.min_cell]
        ys = [ade[g == k].mean() for k in ks]
        ax.plot(ks, ys, "o-", lw=2, color="tab:red")
        for k, yv in zip(ks, ys):
            ax.text(k, yv, f"{(g == k).sum()}", fontsize=7, ha="center", va="bottom", color="gray")
        ax.set_xlabel(name)
        ax.set_ylabel("minADE (m)")
        ax.set_title(f"{name} vs ADE")
        ax.grid(alpha=.3)

    names = list(scores.keys())
    vals = [scores[nm] for nm in names]
    order = np.argsort(vals)[::-1]
    axes[1, 1].barh([names[i] for i in order], [vals[i] for i in order],
                    color="tab:green", alpha=.85)
    axes[1, 1].set_xlabel("η²（方差解释比）")
    axes[1, 1].set_title("哪种「时长」最能解释 ADE")
    axes[1, 1].grid(axis="x", alpha=.3)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    path = OUT_DIR / f"fig_occlusion_duration{tag}.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("-" * 66)
    print(f"已保存（仅此一个文件）: {path}")


if __name__ == "__main__":
    main()
