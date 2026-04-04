"""跑 100 次模拟，统计平均结果"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from talent_vs_luck import TalentVsLuckSimulation

N_RUNS = 100

# 每次运行的统计量
all_max_capital = []
all_median_capital = []
all_mean_capital = []
all_top20_shares = []
all_top10_shares = []
all_top1_shares = []

# Top 5 天赋 & 财富排名相关
top5_richest_talents = []      # 每次首富的天赋
top1_talent_rank = []          # 每次最有天赋的人的财富排名
top5_talent_capitals = []      # 每次最有天赋的人的资本

# 事件命中
top100_net_lucky = []          # 前100名净幸运次数
bot100_net_lucky = []          # 后100名净幸运次数
top100_avg_talent = []         # 前100名平均天赋
bot100_avg_talent = []         # 后100名平均天赋

for i in range(N_RUNS):
    sim = TalentVsLuckSimulation(
        n_agents=1000, world_size=40.0,
        n_lucky_events=50, n_unlucky_events=50,
        perception_radius=1.5, max_steps=80, seed=None,
    )
    history = sim.run()
    final = history[-1]

    all_max_capital.append(final.max())
    all_median_capital.append(np.median(final))
    all_mean_capital.append(final.mean())

    sorted_cap = np.sort(final)[::-1]
    total = sorted_cap.sum()
    n = len(final)
    all_top20_shares.append(sorted_cap[:n//5].sum() / total * 100)
    all_top10_shares.append(sorted_cap[:n//10].sum() / total * 100)
    all_top1_shares.append(sorted_cap[:max(1, n//100)].sum() / total * 100)

    # 首富天赋
    richest_idx = np.argmax(final)
    top5_richest_talents.append(sim.talent[richest_idx])

    # 最有天赋的人的财富排名和资本
    most_talented_idx = np.argmax(sim.talent)
    cap_rank = np.where(np.argsort(final)[::-1] == most_talented_idx)[0][0] + 1
    top1_talent_rank.append(cap_rank)
    top5_talent_capitals.append(final[most_talented_idx])

    # 前/后 100 名
    rank = np.argsort(final)[::-1]
    top100 = rank[:100]
    bot100 = rank[-100:]
    net_lucky = sim.lucky_hits_log.astype(float) - sim.unlucky_hits_log.astype(float)

    top100_net_lucky.append(net_lucky[top100].mean())
    bot100_net_lucky.append(net_lucky[bot100].mean())
    top100_avg_talent.append(sim.talent[top100].mean())
    bot100_avg_talent.append(sim.talent[bot100].mean())

    if (i + 1) % 10 == 0:
        print(f"  Completed {i+1}/{N_RUNS} runs...")

# ============================================================
# 输出统计
# ============================================================
print("\n" + "=" * 65)
print(f"  AGGREGATE RESULTS OVER {N_RUNS} RUNS")
print("=" * 65)

print("\n--- Capital Statistics (avg over 100 runs) ---")
print(f"  Max capital:    {np.mean(all_max_capital):>14.2f}  (std: {np.std(all_max_capital):.2f})")
print(f"  Median capital: {np.mean(all_median_capital):>14.4f}  (std: {np.std(all_median_capital):.4f})")
print(f"  Mean capital:   {np.mean(all_mean_capital):>14.2f}  (std: {np.std(all_mean_capital):.2f})")

print("\n--- Wealth Concentration (avg over 100 runs) ---")
print(f"  Top 20% own: {np.mean(all_top20_shares):.1f}%  (std: {np.std(all_top20_shares):.1f}%)")
print(f"  Top 10% own: {np.mean(all_top10_shares):.1f}%  (std: {np.std(all_top10_shares):.1f}%)")
print(f"  Top  1% own: {np.mean(all_top1_shares):.1f}%  (std: {np.std(all_top1_shares):.1f}%)")

print("\n--- The Richest Agent ---")
print(f"  Avg talent of #1 richest: {np.mean(top5_richest_talents):.4f}  (std: {np.std(top5_richest_talents):.4f})")
print(f"  Population avg talent:    0.6000")
print(f"  => 首富天赋仅略高于平均，远非最高")

print("\n--- The Most Talented Agent ---")
print(f"  Avg wealth rank:    {np.mean(top1_talent_rank):.1f} / 1000  (std: {np.std(top1_talent_rank):.1f})")
print(f"  Avg capital:        {np.mean(top5_talent_capitals):.2f}  (std: {np.std(top5_talent_capitals):.2f})")
print(f"  => 最有天赋的人平均只排在第 {np.mean(top1_talent_rank):.0f} 名")

print("\n--- Top 100 vs Bottom 100 ---")
print(f"  Top  100 avg talent:    {np.mean(top100_avg_talent):.4f}  (vs overall 0.6000)")
print(f"  Bottom 100 avg talent:  {np.mean(bot100_avg_talent):.4f}")
print(f"  Top  100 avg net lucky: {np.mean(top100_net_lucky):+.2f}")
print(f"  Bottom 100 avg net lucky: {np.mean(bot100_net_lucky):+.2f}")

print("=" * 65)

# ============================================================
# 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(f"Talent vs Luck — Aggregate Over {N_RUNS} Runs", fontsize=14, fontweight="bold")

# 1. 首富天赋分布
ax = axes[0, 0]
ax.hist(top5_richest_talents, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(np.mean(top5_richest_talents), color="red", linestyle="--",
           label=f"Mean={np.mean(top5_richest_talents):.3f}")
ax.axvline(0.6, color="green", linestyle=":", label="Population avg=0.600")
ax.set_xlabel("Talent of Richest Agent")
ax.set_ylabel("Count")
ax.set_title("Talent Distribution of #1 Richest (100 runs)")
ax.legend()

# 2. 最有天赋的人的财富排名分布
ax = axes[0, 1]
ax.hist(top1_talent_rank, bins=40, color="darkorange", edgecolor="white", alpha=0.85)
ax.axvline(np.mean(top1_talent_rank), color="red", linestyle="--",
           label=f"Mean rank={np.mean(top1_talent_rank):.0f}")
ax.set_xlabel("Wealth Rank of Most Talented Agent")
ax.set_ylabel("Count")
ax.set_title("Wealth Rank of #1 Most Talented (100 runs)")
ax.legend()

# 3. Top 100 vs Bottom 100 净幸运次数
ax = axes[1, 0]
x = np.arange(N_RUNS)
ax.scatter(x, top100_net_lucky, s=8, alpha=0.6, color="green", label="Top 100")
ax.scatter(x, bot100_net_lucky, s=8, alpha=0.6, color="red", label="Bottom 100")
ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
ax.axhline(np.mean(top100_net_lucky), color="green", linestyle="--", alpha=0.7,
           label=f"Top 100 avg={np.mean(top100_net_lucky):+.1f}")
ax.axhline(np.mean(bot100_net_lucky), color="red", linestyle="--", alpha=0.7,
           label=f"Bottom 100 avg={np.mean(bot100_net_lucky):+.1f}")
ax.set_xlabel("Run #")
ax.set_ylabel("Avg Net Lucky Hits")
ax.set_title("Net Lucky Hits: Top 100 vs Bottom 100")
ax.legend(fontsize=8)

# 4. Top 100 vs Bottom 100 平均天赋
ax = axes[1, 1]
ax.hist(top100_avg_talent, bins=30, color="green", edgecolor="white", alpha=0.7, label="Top 100")
ax.hist(bot100_avg_talent, bins=30, color="red", edgecolor="white", alpha=0.7, label="Bottom 100")
ax.axvline(np.mean(top100_avg_talent), color="green", linestyle="--",
           label=f"Top 100 mean={np.mean(top100_avg_talent):.3f}")
ax.axvline(np.mean(bot100_avg_talent), color="red", linestyle="--",
           label=f"Bottom 100 mean={np.mean(bot100_avg_talent):.3f}")
ax.set_xlabel("Average Talent")
ax.set_ylabel("Count")
ax.set_title("Avg Talent: Top 100 vs Bottom 100 (100 runs)")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("talent_vs_luck_100runs.png", dpi=150, bbox_inches="tight")
print("\nFigure saved to talent_vs_luck_100runs.png")
