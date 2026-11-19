# -*- coding: utf-8 -*-
"""第 5 步：数据体检。

用法：py -3.9 inspect_data.py
可在任意目录运行 —— 脚本按自身位置定位 data\ 与输出图，不依赖当前工作目录。
"""

import glob
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 不弹窗，直接存图
import matplotlib.pyplot as plt

# 让中文标题能正常显示（否则一堆 Glyph missing 警告，图上全是方块）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent      # 本脚本所在目录 = 仓库根目录


def output_dir(*parts):
    """输出写到 项目根\03-结果\<子目录>；若项目结构不存在，退回脚本目录。"""
    root = BASE.parents[1] / "03-结果"
    if not root.is_dir():
        root = BASE
    d = root.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def find_npz():
    pattern = str(BASE / "data" / "**" / "*.npz")
    return sorted(glob.glob(pattern, recursive=True))


def main():
    paths = find_npz()
    print("=" * 66)
    print(f"在 data/ 下找到 {len(paths)} 个 .npz 文件")
    for p in paths[:12]:
        print(f"  {os.path.getsize(p) / 1e6:8.2f} MB  {p}")
    print("=" * 66)

    if not paths:
        print("没找到数据。先运行： py -3.9 data/_download_data.py")
        return

    # 优先级：egotraj 的 test > 任意 test > 第一个
    def pick(pattern):
        hits = [p for p in paths if all(k in p.lower() for k in pattern)]
        return sorted(hits)[0] if hits else None

    PATH = pick(("egotraj", "test")) or pick(("test",)) or paths[0]
    print("使用文件:", PATH)

    d = np.load(PATH)
    print("包含的键:", list(d.files))

    if "all_obs" not in d.files:
        print("!! 这个 npz 里没有 all_obs，可能下错了层级（要 L2-processed）")
        return

    obs, pred = d["all_obs"], d["all_pred"]

    print("-" * 66)
    print("all_obs 形状 :", obs.shape, " (期望 [N, 8, 7])")
    print("all_pred 形状:", pred.shape, " (期望 [N, 20, 7])")

    assert obs.shape[1:] == (8, 7), f"all_obs 形状不符: {obs.shape}"
    assert pred.shape[1:] == (20, 7), f"all_pred 形状不符: {pred.shape}"
    print("形状断言通过，agent 总数 N =", obs.shape[0])

    if "num_peds" in d.files:
        print("num_peds 形状:", d["num_peds"].shape,
              "| 最少/最多 agent:", int(d["num_peds"].min()), "/", int(d["num_peds"].max()))
    if "seq_start_end" in d.files:
        print("seq_start_end 形状:", d["seq_start_end"].shape,
              "| 序列数:", d["seq_start_end"].shape[0])

    # ---- 遮挡率 ----
    mask = obs[:, :, 5]
    miss = 1 - mask.mean(axis=1)

    print("-" * 66)
    print("遮挡率（1 - valid_mask 均值）分位：")
    for q in [0, .25, .5, .75, .9, 1.0]:
        print(f"  {int(q * 100):>3}%  ->  {np.quantile(miss, q):.3f}")
    print(f"  完全无缺失 : {(miss == 0).mean():.3f}")
    print(f"  缺失 > 10% : {(miss > .1).mean():.3f}")
    print(f"  缺失 > 30% : {(miss > .3).mean():.3f}")

    # ---- 画 10 条样本 ----
    rng = np.random.default_rng(0)
    idx = rng.choice(len(obs), size=min(10, len(obs)), replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, i in zip(axes.ravel(), idx):
        ax.plot(pred[i, :8, 0], pred[i, :8, 1], "o-", ms=3, lw=1,
                color="tab:blue", label="BEV 干净历史")
        ax.plot(pred[i, 8:, 0], pred[i, 8:, 1], "s--", ms=3, lw=1,
                color="tab:green", label="BEV 干净未来")
        ax.plot(obs[i, :, 0], obs[i, :, 1], "x-", ms=4, lw=1,
                color="tab:red", label="FPV 噪声历史")
        ax.set_title(f"#{i}  miss={miss[i]:.2f}", fontsize=9)
        ax.axis("equal")
        ax.grid(alpha=.3)

    axes[0, 0].legend(fontsize=7, loc="best")
    plt.tight_layout()
    out = output_dir("图片") / "fig_samples.png"
    plt.savefig(str(out), dpi=160, bbox_inches="tight")
    print("-" * 66)
    print("已保存图片:", out)
    print("=" * 66)
    print("数据体检完成。")


if __name__ == "__main__":
    main()
