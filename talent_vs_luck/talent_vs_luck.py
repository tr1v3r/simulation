"""
Talent vs Luck: The Role of Randomness in Success and Failure
基于 Pluchino, Biondo & Rapisarda (2018) 论文的 Agent-Based Model 模拟

模型规则:
1. N 个 Agent 随机分布在一个 L×L 的二维世界中
2. 每个 Agent 的天赋 Ti ~ N(0.6, 0.1)，截断到 [0, 1]
3. 所有 Agent 初始资本相同 (C0 = 10)
4. 每个时间步随机生成 Nh 个幸运事件和 Nu 个不幸事件
5. 事件在 Agent 感知半径内触发交互:
   - 幸运事件: 以概率 Ti 抓住机会, 资本翻倍; 否则不变
   - 不幸事件: 无条件资本减半
6. 一个 Agent 在同一步可被多个事件影响 (可叠加)
7. 模拟 M 个时间步 (代表 40 年工作生涯) 后统计最终资本分布
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 模拟核心
# ============================================================

class TalentVsLuckSimulation:
    def __init__(
        self,
        n_agents: int = 1000,
        world_size: float = 40.0,
        n_lucky_events: int = 50,
        n_unlucky_events: int = 50,
        perception_radius: float = 1.5,
        talent_mean: float = 0.6,
        talent_std: float = 0.1,
        initial_capital: float = 10.0,
        max_steps: int = 80,
        seed: int | None = None,
    ):
        self.n_agents = n_agents
        self.L = world_size
        self.n_lucky = n_lucky_events
        self.n_unlucky = n_unlucky_events
        self.radius = perception_radius
        self.max_steps = max_steps

        self.rng = np.random.default_rng(seed)

        # Agent 属性
        self.agent_pos = self.rng.uniform(0, self.L, size=(n_agents, 2))
        self.talent = np.clip(
            self.rng.normal(talent_mean, talent_std, size=n_agents), 0.0, 1.0
        )
        self.capital = np.full(n_agents, initial_capital, dtype=np.float64)

        # 每步每个 Agent 记录遇到多少幸运/不幸事件 (用于调试)
        self.lucky_hits_log = np.zeros(n_agents, dtype=int)
        self.unlucky_hits_log = np.zeros(n_agents, dtype=int)

        self.history = [self.capital.copy()]

    def _generate_events(self):
        """每步重新随机生成事件位置"""
        lucky_pos = self.rng.uniform(0, self.L, size=(self.n_lucky, 2))
        unlucky_pos = self.rng.uniform(0, self.L, size=(self.n_unlucky, 2))
        return lucky_pos, unlucky_pos

    def _check_interactions(self, lucky_pos: np.ndarray, unlucky_pos: np.ndarray):
        """检测事件与 Agent 的交互"""
        r2 = self.radius ** 2

        # 幸运事件: 向量化计算所有 event-agent 距离
        # lucky_pos: (Nh, 2), agent_pos: (N, 2)
        # 计算每个 event 到所有 agent 的距离
        lucky_hits = np.zeros(self.n_agents, dtype=int)
        for ep in lucky_pos:
            dists_sq = np.sum((self.agent_pos - ep) ** 2, axis=1)
            hit_mask = dists_sq < r2
            if np.any(hit_mask):
                # 以概率 Ti 决定是否抓住机会
                rolls = self.rng.random(np.sum(hit_mask))
                success = rolls < self.talent[hit_mask]
                self.capital[hit_mask] = np.where(
                    success, self.capital[hit_mask] * 2, self.capital[hit_mask]
                )
                lucky_hits[hit_mask] += 1

        # 不幸事件: 无条件减半
        unlucky_hits = np.zeros(self.n_agents, dtype=int)
        for ep in unlucky_pos:
            dists_sq = np.sum((self.agent_pos - ep) ** 2, axis=1)
            hit_mask = dists_sq < r2
            if np.any(hit_mask):
                self.capital[hit_mask] /= 2.0
                unlucky_hits[hit_mask] += 1

        self.lucky_hits_log += lucky_hits
        self.unlucky_hits_log += unlucky_hits

    def run(self) -> np.ndarray:
        """运行完整模拟, 返回每个 Agent 每一步的资本历史"""
        for step in range(1, self.max_steps + 1):
            lucky_pos, unlucky_pos = self._generate_events()
            self._check_interactions(lucky_pos, unlucky_pos)
            self.history.append(self.capital.copy())
        return np.array(self.history)


# ============================================================
# 可视化
# ============================================================

def plot_results(sim: TalentVsLuckSimulation, history: np.ndarray):
    """生成论文风格的结果图"""
    final = history[-1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Talent vs Luck Simulation\n"
        "(Pluchino, Biondo & Rapisarda, 2018)",
        fontsize=14, fontweight="bold",
    )

    # --- 1. 天赋分布 ---
    ax = axes[0, 0]
    ax.hist(sim.talent, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Talent")
    ax.set_ylabel("Count")
    ax.set_title("Talent Distribution (Normal)")
    ax.axvline(sim.talent.mean(), color="red", linestyle="--", label=f"Mean={sim.talent.mean():.2f}")
    ax.legend()

    # --- 2. 最终资本分布 (log-log 直方图) ---
    ax = axes[0, 1]
    positive = final[final > 0]
    log_bins = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 50)
    counts, edges, patches = ax.hist(positive, bins=log_bins, color="darkorange",
                                      edgecolor="white", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("Final Capital (log scale)")
    ax.set_ylabel("Count")
    ax.set_title("Final Capital Distribution (Pareto-like)")
    ax.axvline(np.median(positive), color="red", linestyle="--",
               label=f"Median={np.median(positive):.4f}")
    ax.axvline(10.0, color="green", linestyle=":", label="Initial capital=10")
    ax.legend(fontsize=8)

    # --- 3. 天赋 vs 最终资本 ---
    ax = axes[1, 0]
    sc = ax.scatter(sim.talent, final, c=np.log10(final + 1), cmap="viridis",
                    s=8, alpha=0.6, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="log10(Capital)")
    top_idx = np.argsort(final)[-5:]
    ax.scatter(sim.talent[top_idx], final[top_idx], c="red", s=60, marker="*",
               zorder=5, label="Top 5 richest")
    ax.set_xlabel("Talent")
    ax.set_ylabel("Final Capital")
    ax.set_title("Talent vs Final Capital")
    ax.legend(fontsize=8)

    # --- 4. 各财富层级的天赋均值 ---
    ax = axes[1, 1]
    sorted_cap = np.sort(final)[::-1]
    total = sorted_cap.sum()
    top20_share = sorted_cap[: len(sorted_cap) // 5].sum() / total * 100

    percentiles = [80, 90, 95, 99]
    talents_at_percentiles = {}
    for p in percentiles:
        threshold = np.percentile(final, p)
        mask = final >= threshold
        talents_at_percentiles[f"Top {100-p}%"] = sim.talent[mask].mean()

    labels = list(talents_at_percentiles.keys())
    vals = list(talents_at_percentiles.values())
    bars = ax.bar(labels, vals, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
    ax.set_ylabel("Average Talent")
    ax.set_title(f"Avg Talent by Wealth Tier  (Top 20% own {top20_share:.1f}% wealth)")
    ax.set_ylim(0.4, 0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("talent_vs_luck_results.png", dpi=150, bbox_inches="tight")
    print("Figure saved to talent_vs_luck_results.png")
    plt.close()


def plot_capital_evolution(sim: TalentVsLuckSimulation, history: np.ndarray, top_n: int = 10):
    """绘制资本随时间的变化曲线"""
    final = history[-1]
    top_idx = np.argsort(final)[-top_n:]

    fig, ax = plt.subplots(figsize=(12, 6))
    steps = np.arange(history.shape[0])
    for rank, idx in enumerate(reversed(top_idx)):
        ax.plot(steps, history[:, idx], alpha=0.7,
                label=f"#{rank+1} (T={sim.talent[idx]:.3f})")
    ax.set_xlabel("Time Step (each ~ 6 months)")
    ax.set_ylabel("Capital")
    ax.set_yscale("linear")
    ax.set_title(f"Capital Evolution of Top {top_n} Agents")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig("talent_vs_luck_evolution.png", dpi=150, bbox_inches="tight")
    print("Figure saved to talent_vs_luck_evolution.png")
    plt.close()


def print_statistics(sim: TalentVsLuckSimulation, history: np.ndarray):
    """打印统计摘要"""
    final = history[-1]

    print("=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"Agents: {sim.n_agents}  |  Steps: {sim.max_steps}")
    print(f"World size: {sim.L}x{sim.L}  |  Perception radius: {sim.radius}")
    print(f"Lucky events/step: {sim.n_lucky}  |  Unlucky events/step: {sim.n_unlucky}")
    print()

    # 资本统计
    print("--- Capital Statistics ---")
    print(f"  Mean:   {final.mean():.2f}")
    print(f"  Median: {np.median(final):.2f}")
    print(f"  Std:    {final.std():.2f}")
    print(f"  Min:    {final.min():.4f}")
    print(f"  Max:    {final.max():.2f}")
    print()

    # Pareto 法则
    sorted_cap = np.sort(final)[::-1]
    total = sorted_cap.sum()
    for pct in [0.20, 0.10, 0.01]:
        n = max(1, int(pct * len(final)))
        share = sorted_cap[:n].sum() / total * 100
        print(f"  Top {pct*100:.0f}% agents own {share:.1f}% of total capital")
    print()

    # 最富 vs 最有天赋
    top5_capital = np.argsort(final)[-5:]
    top5_talent = np.argsort(sim.talent)[-5:]

    print("--- Top 5 Richest ---")
    for i, idx in enumerate(reversed(top5_capital)):
        rank = i + 1
        l_hits = sim.lucky_hits_log[idx]
        u_hits = sim.unlucky_hits_log[idx]
        print(f"  #{rank}: Capital={final[idx]:12.2f}  Talent={sim.talent[idx]:.4f}"
              f"  (Lucky hits={l_hits}, Unlucky hits={u_hits})")

    print()
    print("--- Top 5 Most Talented ---")
    for i, idx in enumerate(reversed(top5_talent)):
        rank = i + 1
        cap_rank = np.where(np.argsort(final)[::-1] == idx)[0][0] + 1
        l_hits = sim.lucky_hits_log[idx]
        u_hits = sim.unlucky_hits_log[idx]
        print(f"  #{rank}: Talent={sim.talent[idx]:.4f}  Capital={final[idx]:12.2f}"
              f"  Wealth Rank #{cap_rank}  (L={l_hits}, U={u_hits})")

    print()

    # 事件命中统计
    print("--- Event Hit Statistics (across all agents) ---")
    print(f"  Avg lucky hits/agent:   {sim.lucky_hits_log.mean():.2f}")
    print(f"  Avg unlucky hits/agent: {sim.unlucky_hits_log.mean():.2f}")

    print("=" * 60)


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    sim = TalentVsLuckSimulation(
        n_agents=1000,
        world_size=40.0,          # 缩小世界, 提高事件密度
        n_lucky_events=50,        # 每步 50 个幸运事件
        n_unlucky_events=50,      # 每步 50 个不幸事件
        perception_radius=1.5,    # 感知半径
        talent_mean=0.6,
        talent_std=0.1,
        initial_capital=10.0,
        max_steps=80,             # 80 步 ≈ 40 年
        seed=None,
    )

    print("Running simulation...")
    history = sim.run()
    print("Done.\n")

    print_statistics(sim, history)
    plot_results(sim, history)
    plot_capital_evolution(sim, history)
