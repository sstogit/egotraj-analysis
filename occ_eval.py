# -*- coding: utf-8 -*-
"""遮挡分层评测器。

对任意模型的预测输出（--save_for_vis 产生的 samples/*.pkl），
按「遮挡」这个维度分层，输出每个档位的误差与 η²，用来回答：
    "模型在丢帧时退化多少？退化是阶跃还是渐进？"

用法（放在 EgoTraj-Bench 根目录）：
    py -3.9 occ_eval.py --tag biflow
    py -3.9 occ_eval.py --samples <某模型的 samples 目录> --tag v2
    py -3.9 occ_eval.py --expect 0.2997        # 开启参照值校验

输出（写到 03-结果\\评测\\<tag>\\）：
    occ_eval_<tag>.csv     逐样本 ade/fde/遮挡特征
    occ_table_<tag>.csv    分档汇总表
    fig_occ_eval_<tag>.png 四联图
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
DEFAULT_SAMPLES = BASE / "checkpoints" / "EgoTraj-TBD" / "samples"
NPZ = BASE.parents[1] / "01-数据" / "L2-processed" / "EgoTraj-TBD-v1.1" / "egotraj_tbd_test.npz"
RESULTS = BASE.parents[1] / "03-结果" / "评测"


def load_predictions(samples_dir):
    """按 batch 编号数值排序——字典序会把 batch_10 排到 batch_2 前面。"""
    files = glob.glob(os.path.join(samples_dir, "visual_mode_best_batch_*.pkl"))
    if not files:
        raise SystemExit(f"没有 pkl：{samples_dir}")
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


def occlusion_features(mask):
    """mask: (N, 8)，1=可见。返回 n_missing / max_run / tail。"""
    miss = (mask == 0)
    n_missing = miss.sum(axis=1)

    def longest(row):
        best = cur = 0
        for v in row:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best

    def trailing(row):
        """末尾连续缺失帧数：从最后一帧往前数 0。"""
        k = 0
        for v in row[::-1]:
            if v == 0:
                k += 1
            else:
                break
        return k

    max_run = np.array([longest(r) for r in miss])
    tail = np.array([trailing(r) for r in mask])      # 传 mask（1=可见），数末尾的 0
    return n_missing, max_run, tail


def eta2(y, g):
    grand = y.mean()
    ss_tot = ((y - grand) ** 2).sum()
    ss_bet = sum(len(y[g == k]) * (y[g == k].mean() - grand) ** 2 for k in np.unique(g))
    return float(ss_bet / ss_tot) if ss_tot > 0 else 0.0


def group_table(name, g, ade, ade_norm, kmax, min_cell):
    rows = []
    print("-" * 70)
    print(f"{name}")
    print(f"{'值':<6}{'n':>7}{'占比':>8}{'minADE':>9}{'ADE/行程':>11}")
    for k in range(kmax + 1):
        m = g == k
        if m.sum() < min_cell:
            continue
        an = np.nanmean(ade_norm[m])
        rows.append(dict(stratum=f"{name}={k}", n=int(m.sum()),
                         share=round(float(m.mean()), 4),
                         ade=round(float(ade[m].mean()), 4),
                         ade_norm=round(float(an), 4),
                         eta2=round(eta2(ade, g), 4)))
        print(f"{k:<6}{m.sum():>7d}{m.mean():>8.3f}{ade[m].mean():>9.4f}{an:>11.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="遮挡分层评测器")
    ap.add_argument("--samples", default=str(DEFAULT_SAMPLES), help="模型的 samples 目录")
    ap.add_argument("--tag", default="biflow", help="输出文件名标签，也是子目录名")
    ap.add_argument("--expect", type=float, default=None,
                    help="参照 minADE（如 0.2997）；给了就校验，偏差 >5e-3 直接退出")
    ap.add_argument("--min-cell", type=int, default=30)
    args = ap.parse_args()

    d = load_predictions(Path(args.samples))
    z = np.load(NPZ)
    mask = z["all_obs"][:, :, 5]

    if len(mask) != len(d["fut_gt"]):
        raise SystemExit(f"数量不一致：mask {len(mask)} vs pred {len(d['fut_gt'])}")

    dist = np.linalg.norm(d["fut_pred"] - d["fut_gt"][:, None], axis=-1)
    ade = dist.mean(axis=-1).min(axis=1)
    fde = dist[:, :, -1].min(axis=1)
    travel = np.linalg.norm(np.diff(d["fut_gt"], axis=1), axis=-1).sum(axis=1)
    still = travel < 0.5
    ade_norm = np.where(still, np.nan, ade / np.maximum(travel, 1e-6))

    print("=" * 70)
    print(f"模型: {args.tag}    样本数: {len(ade)}")
    print(f"minADE = {ade.mean():.4f}   minFDE = {fde.mean():.4f}   静止样本 {still.sum()} 条")
    if args.expect is not None:
        if abs(ade.mean() - args.expect) > 5e-3:
            raise SystemExit(f"与参照值 {args.expect} 不符，先别信后面的表")
        print(f"校验通过（参照 {args.expect}）")

    n_missing, max_run, tail = occlusion_features(mask)

    rows = []
    rows += group_table("缺失帧数", n_missing, ade, ade_norm, 8, args.min_cell)
    rows += group_table("最长连续缺失", max_run, ade, ade_norm, 8, args.min_cell)
    rows += group_table("末尾连续缺失", tail, ade, ade_norm, 8, args.min_cell)

    print("-" * 70)
    print("η²（遮挡特征解释 ADE 的能力）:")
    scores = {"缺失帧数": eta2(ade, n_missing),
              "最长连续": eta2(ade, max_run),
              "末尾连续": eta2(ade, tail)}
    for k, v in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<10}{v:.4f}")

    print("-" * 70)
    print("阶跃检验：无缺失 vs 有缺失")
    a, b = ade[n_missing == 0], ade[n_missing > 0]
    print(f"  无缺失 n={len(a):5d}  minADE={a.mean():.4f}")
    print(f"  有缺失 n={len(b):5d}  minADE={b.mean():.4f}   倍数 {b.mean()/a.mean():.2f}×")

    # ---- 输出 ----
    out = RESULTS / args.tag
    out.mkdir(parents=True, exist_ok=True)

    per = out / f"occ_eval_{args.tag}.csv"
    np.savetxt(per, np.column_stack([ade, fde, np.nan_to_num(ade_norm), n_missing, max_run, tail]),
               delimiter=",", header="ade,fde,ade_norm,n_missing,max_run,missing_tail",
               comments="", fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"])

    tab = out / f"occ_table_{args.tag}.csv"
    with open(tab, "w", encoding="utf-8-sig", newline="") as fh:
        fh.write("stratum,n,share,ade,ade_norm,eta2\n")
        for r in rows:
            fh.write(f"{r['stratum']},{r['n']},{r['share']},{r['ade']},{r['ade_norm']},{r['eta2']}\n")

    # ---- 四联图 ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, g, name in ((axes[0, 0], n_missing, "缺失帧数"),
                        (axes[0, 1], max_run, "最长连续缺失"),
                        (axes[1, 0], tail, "末尾连续缺失")):
        ks = [k for k in range(9) if (g == k).sum() >= args.min_cell]
        ys = [ade[g == k].mean() for k in ks]
        ax.plot(ks, ys, "o-", lw=2, color="tab:red")
        for k, yv in zip(ks, ys):
            ax.text(k, yv, f"{(g == k).sum()}", fontsize=7, ha="center",
                    va="bottom", color="gray")
        ax.set_xlabel(name)
        ax.set_ylabel("minADE (m)")
        ax.set_title(f"{name} vs ADE（{args.tag}）")
        ax.grid(alpha=.3)

    names = list(scores)
    vals = [scores[k] for k in names]
    order = np.argsort(vals)[::-1]
    axes[1, 1].barh([names[i] for i in order], [vals[i] for i in order],
                    color="tab:green", alpha=.85)
    axes[1, 1].set_xlabel("η²")
    axes[1, 1].set_title("遮挡特征的方差解释比")
    axes[1, 1].grid(axis="x", alpha=.3)

    plt.tight_layout()
    fig_path = out / f"fig_occ_eval_{args.tag}.png"
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("-" * 70)
    print("已保存：")
    print("  ", per)
    print("  ", tab)
    print("  ", fig_path)


if __name__ == "__main__":
    main()
