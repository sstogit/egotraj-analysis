# -*- coding: utf-8 -*-
"""第 11 步：从 --save_for_vis 的 pkl 里算逐样本误差，按观测噪声分组，支持多阈值扫描。

用法：
    py -3.9 analyze_per_sample.py                          # 默认扫全部阈值
    py -3.9 analyze_per_sample.py --clean 1e-6 1e-3 0.1    # 只扫指定几个
    py -3.9 analyze_per_sample.py --fde ade_best           # 换 FDE 口径
    py -3.9 analyze_per_sample.py --no-dedupe              # 分组相同也照样写文件

后缀规则：所有输出名 = <主干>_<fde口径>_<阈值>.<扩展名>
    阈值统一格式化成科学计数法：1e-06 / 1e-03 / 5e-02 / 1e-01 ...

指标口径（和官方评测对齐，已校验）：
    minADE = 每条样本在 K 条候选里取「整条最准」的那条，算平均位移
    minFDE = 每条样本在 K 条候选里取「终点最准」的那条，算终点位移
    两者不一定来自同一条候选。

分组方式：
    本数据的噪声分布是双峰的（约 70% 精确为 0），等分位会退化，
    所以先切出「干净」样本，再把剩下的噪声样本按四分位细分。
"""

import argparse
import csv
import glob
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent      # 本脚本所在目录 = 仓库根目录
SAMPLE_DIR = BASE / "checkpoints" / "EgoTraj-TBD" / "samples"


def output_dir(*parts):
    """输出写到 项目根\03-结果\<子目录>；若项目结构不存在，退回脚本目录。"""
    root = BASE.parents[1] / "03-结果"
    if not root.is_dir():
        root = BASE
    d = root.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


KEYS = ["fut_gt", "fut_pred", "hist_obs", "hist_gt"]

# 跨模式汇总表的列（一张表累加所有跑过的 口径×阈值）
SUMMARY_COLS = ["fde", "clean_tag", "clean_eps", "n_clean", "clean_share",
                "clean_ade", "clean_fde", "q1_ade", "q2_ade", "q3_ade", "q4_ade", "q4_fde",
                "ratio_q4_clean", "overall_ade", "overall_fde", "run_at"]


def upsert_summary(rows):
    """把本次结果并入 03-结果\\summary.csv：同一 (fde, 阈值) 覆盖旧行，其余保留。"""
    path = output_dir() / "summary.csv"

    old = []
    if path.exists():
        with open(path, newline="", encoding="utf-8-sig") as fh:
            old = [dict(r) for r in csv.DictReader(fh)]

    def key(d):
        return (d.get("fde", ""), d.get("clean_tag", ""))

    fresh = {key(r) for r in rows}
    merged = [r for r in old if key(r) not in fresh] + rows
    merged.sort(key=lambda d: (d.get("fde", ""), float(d.get("clean_eps") or 0)))

    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLS)
        w.writeheader()
        for d in merged:
            w.writerow({k: d.get(k, "") for k in SUMMARY_COLS})

    return path, merged

# 默认扫描的干净阈值（噪声 ≤ 该值算「无观测噪声」）
# 1e-6 放第一位：它是主结果，后面几个是稳健性扫描
DEFAULT_CLEANS = [1e-6, 1e-9, 1e-12, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 3e-1]


def fmt_eps(eps):
    """阈值 → 文件名安全的科学计数法，如 1e-06、5e-02、1e-01。"""
    return f"{eps:.0e}"


def load_pkls():
    files = sorted(glob.glob(os.path.join(SAMPLE_DIR, "visual_mode_best_batch_*.pkl")))
    print(f"找到 {len(files)} 个 pkl")
    if not files:
        raise SystemExit("无 pkl")

    buf = {k: [] for k in KEYS}
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        for k in KEYS:
            if k not in d or d[k] is None:
                raise SystemExit(f"pkl 缺少 {k}")
            buf[k].append(np.asarray(d[k], dtype=np.float32))
    return {k: np.concatenate(v, axis=0) for k, v in buf.items()}


# ─────────────────────────────────────────────────────────────
# FDE 口径注册表
# 加一个新模式 = 写一个函数 + 在 FDE_MODES 里加一行，CLI 自动支持
# 签名统一：(ade_per_k, fde_per_k, n) -> (n,) 的数组
# ─────────────────────────────────────────────────────────────

def fde_official(ade_per_k, fde_per_k, n):
    """论文口径：每条样本的 K 条候选里，终点各自取最小。"""
    return fde_per_k.min(axis=1)


def fde_ade_best(ade_per_k, fde_per_k, n):
    """实验口径：先按 ADE 选出整条最准的候选，再取它的终点。"""
    return fde_per_k[np.arange(n), ade_per_k.argmin(axis=1)]


def fde_first(ade_per_k, fde_per_k, n):
    """基线：永远取第 0 条候选的终点（不做任何挑选）。"""
    return fde_per_k[:, 0]


FDE_MODES = {
    # 名字: (说明, 函数)
    "official": ("终点单独取最小（论文口径，校验基准）", fde_official),
    "ade_best": ("ADE 最优候选的终点", fde_ade_best),
    "first":    ("第 0 条候选的终点（不挑选的基线）", fde_first),
}


def main():
    ap = argparse.ArgumentParser(description="EgoTraj-Bench 分层分析（多阈值扫描）")
    ap.add_argument("--fde", choices=list(FDE_MODES), default="official",
                    help="FDE 取法；可选：" +
                         "；".join(f"{k}={v[0]}" for k, v in FDE_MODES.items()))
    ap.add_argument("--clean", nargs="+", type=float, default=DEFAULT_CLEANS,
                    help="干净阈值，可给多个；默认：" + " ".join(fmt_eps(e) for e in DEFAULT_CLEANS))
    ap.add_argument("--no-dedupe", action="store_true",
                    help="分组与之前某个阈值完全相同时，仍然照写文件（默认跳过）")
    args = ap.parse_args()

    fde_mode = args.fde

    d = load_pkls()
    gt, pred = d["fut_gt"], d["fut_pred"]
    hist_obs, hist_gt = d["hist_obs"], d["hist_gt"]

    n, K = pred.shape[0], pred.shape[1]
    print("-" * 72)
    print(f"样本数 N = {n}，候选数 K = {K}")

    dist = np.linalg.norm(pred - gt[:, None, :, :], axis=-1)   # [N, K, 12]
    ade_per_k = dist.mean(axis=-1)
    fde_per_k = dist[:, :, -1]

    ade = ade_per_k.min(axis=1)

    # 把每种口径都算一遍，列出全局均值，方便对比
    print("FDE 口径一览：")
    for name, (desc, fn) in FDE_MODES.items():
        mark = "   ← 本次使用" if name == fde_mode else ""
        print(f"  {name:<10}{fn(ade_per_k, fde_per_k, n).mean():.4f}   {desc}{mark}")

    fde = FDE_MODES[fde_mode][1](ade_per_k, fde_per_k, n)
    fde_official_vals = FDE_MODES["official"][1](ade_per_k, fde_per_k, n)

    print(f"逐样本 minADE 均值 = {ade.mean():.4f}   （官方 0.2997）")

    ok = abs(ade.mean() - 0.2997) < 5e-3 and abs(fde_official_vals.mean() - 0.4403) < 5e-3
    print("校验:", "通过" if ok else "不通过", "（只校验论文口径）")
    if not ok:
        raise SystemExit("复现结果与原论文不符")

    same = (ade_per_k.argmin(axis=1) == fde_per_k.argmin(axis=1)).mean()
    print(f"argmin(ADE) 与 argmin(FDE) 选中同一条候选的比例 = {same:.3f}")

    noise = np.linalg.norm(hist_obs - hist_gt, axis=-1).mean(axis=1)
    print(f"噪声均值 = {noise.mean():.4f} m，中位数 = {np.median(noise):.4f} m")
    print(f"非零噪声最小值 = {noise[noise > 0].min():.4f} m" if (noise > 0).any() else "")
    print("-" * 72)
    print(f"阈值扫描：共 {len(args.clean)} 个 —— " + " ".join(fmt_eps(e) for e in args.clean))

    labels = ["干净", "噪声Q1", "噪声Q2", "噪声Q3", "噪声Q4"]
    seen = []          # [(eps, is_clean)]，用于去重
    written = []       # 实际写出文件的阈值
    summary_rows = []  # 汇总到 summary.csv 的行

    for eps in args.clean:
        tag = fmt_eps(eps)
        is_clean = noise <= eps
        mask_key = is_clean

        dup = next((e for e, m in seen if np.array_equal(m, mask_key)), None)
        if dup is not None and not args.no_dedupe:
            print(f"  阈值 {tag:<8} 干净 {is_clean.sum():>5} 条  —— 与 {fmt_eps(dup)} 分组相同，跳过")
            continue
        seen.append((eps, mask_key))

        rest = noise[~is_clean]
        if len(rest) < 40:
            print(f"  阈值 {tag:<8} 噪声样本仅剩 {len(rest)} 条，跳过")
            continue

        edges = np.quantile(rest, [0, .25, .5, .75, 1.0])
        edges[0] -= 1e-9
        edges[-1] += 1e-9

        bucket = np.zeros(n, dtype=int)
        bucket[~is_clean] = 1 + np.clip(np.digitize(rest, edges[1:-1]), 0, 3)

        rows = []
        total = ade.sum()
        for i in range(5):
            m = bucket == i
            if m.sum() == 0:
                continue
            rows.append(dict(n=int(m.sum()), noise=noise[m].mean(), ade=ade[m].mean(),
                             fde=fde[m].mean(), share=m.mean(), contrib=ade[m].sum() / total))

        print("-" * 72)
        print(f"阈值 ≤ {eps:g}  ({tag})   干净样本 {is_clean.sum()} 条 ({is_clean.mean():.1%})")
        print(f"{'分组':<8}{'n':>6}{'占比':>8}{'噪声':>9}{'minADE':>9}{'minFDE':>9}{'误差占比':>10}")
        for i, r in enumerate(rows):
            print(f"{labels[i]:<8}{r['n']:>6}{r['share']:>8.3f}{r['noise']:>9.4f}"
                  f"{r['ade']:>9.4f}{r['fde']:>9.4f}{r['contrib']:>10.3f}")
        if len(rows) >= 2:
            print(f"最吵的一档 vs 干净样本：minADE 高 {rows[-1]['ade'] / rows[0]['ade']:.2f} 倍")

        suffix = f"_{fde_mode}_{tag}"

        npz_path = output_dir("数据明细") / f"per_sample{suffix}.npz"
        np.savez(npz_path, ade=ade, fde=fde, noise=noise, bucket=bucket)

        csv_path = output_dir("数据明细") / f"per_sample{suffix}.csv"
        np.savetxt(csv_path,
                   np.column_stack([ade, fde, noise, bucket]),
                   delimiter=",", header="ade,fde,noise,bucket", comments="",
                   fmt=["%.6f", "%.6f", "%.6f", "%d"])

        # ---- 画图 ----
        fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
        x = np.arange(len(rows))
        names = labels[:len(rows)]

        axes[0].hist(np.clip(noise, 0, 3), bins=60, color="tab:gray", alpha=.85)
        axes[0].set_xlabel("观测噪声 (m，截断到 3)")
        axes[0].set_ylabel("样本数")
        axes[0].set_title(f"噪声分布（≤{eps:g} 记为干净）")
        axes[0].grid(alpha=.3)

        axes[1].bar(x, [r["ade"] for r in rows], color="tab:blue", alpha=.85)
        axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=20)
        axes[1].set_ylabel("minADE (m)"); axes[1].set_title(f"噪声档位 vs minADE（≤{eps:g}）")
        axes[1].grid(axis="y", alpha=.3)

        axes[2].bar(x, [r["fde"] for r in rows], color="tab:orange", alpha=.85)
        axes[2].set_xticks(x); axes[2].set_xticklabels(names, rotation=20)
        axes[2].set_ylabel("minFDE (m)"); axes[2].set_title(f"噪声档位 vs minFDE（≤{eps:g}）")
        axes[2].grid(axis="y", alpha=.3)

        axes[3].scatter(noise, ade, s=2, alpha=.12, color="tab:gray")
        axes[3].plot([r["noise"] for r in rows], [r["ade"] for r in rows],
                     "o-", color="tab:red", lw=2, label="分组均值")
        axes[3].set_xlim(-0.05, 3)
        axes[3].set_xlabel("观测噪声 (m)"); axes[3].set_ylabel("minADE (m)")
        axes[3].set_title(f"逐样本散点（≤{eps:g}）"); axes[3].legend(); axes[3].grid(alpha=.3)

        plt.tight_layout()
        fig_path = output_dir("图片") / f"fig_bucket{suffix}.png"
        plt.savefig(fig_path, dpi=180, bbox_inches="tight")
        plt.close(fig)          # 关键：循环里必须关掉，否则图越积越多

        written.append(tag)
        print(f"  已保存 {tag}: {fig_path.name} / {npz_path.name} / {csv_path.name}")

        def r4(i, field):
            return round(float(rows[i][field]), 4) if len(rows) > i else ""

        summary_rows.append({
            "fde": fde_mode,
            "clean_tag": tag,
            "clean_eps": f"{eps:.0e}",
            "n_clean": int(is_clean.sum()),
            "clean_share": round(float(is_clean.mean()), 4),
            "clean_ade": r4(0, "ade"),
            "clean_fde": r4(0, "fde"),
            "q1_ade": r4(1, "ade"),
            "q2_ade": r4(2, "ade"),
            "q3_ade": r4(3, "ade"),
            "q4_ade": r4(4, "ade"),
            "q4_fde": r4(4, "fde"),
            "ratio_q4_clean": (round(float(rows[-1]["ade"] / rows[0]["ade"]), 2)
                               if len(rows) > 1 else ""),
            "overall_ade": round(float(ade.mean()), 4),
            "overall_fde": round(float(fde.mean()), 4),
            "run_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })

    print("-" * 72)
    print(f"完成：{len(written)} 组输出 —— " + " ".join(written))

    if summary_rows:
        path, merged = upsert_summary(summary_rows)
        print("-" * 72)
        print(f"跨模式汇总 → {path}")
        print(f"{'fde':<10}{'eps':<9}{'n_clean':>8}{'干净ADE':>9}{'干净FDE':>9}"
              f"{'Q4_ADE':>9}{'Q4_FDE':>9}{'比值':>7}")
        for d in merged:
            print(f"{d['fde']:<10}{d['clean_tag']:<9}{d['n_clean']:>8}{d['clean_ade']:>9}"
                  f"{d['clean_fde']:>9}{d['q4_ade']:>9}{d['q4_fde']:>9}"
                  f"{d['ratio_q4_clean']:>7}")


if __name__ == "__main__":
    main()
