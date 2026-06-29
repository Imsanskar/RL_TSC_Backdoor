"""
Reward Comparison Evaluator
===========================
Runs two versions of the attacker side-by-side:
  - OLD: reward = -delay - lambda * fake_count
  - NEW: reward = (delay_t - delay_{t-1}) - lambda * fake_count  (delta delay)

Both use identical world resets and random seeds so results are comparable.
Outputs a matplotlib plot of average travel time per episode for both versions.

Usage:
    python evaluate_reward_comparison.py --config configs/tsc_rl_adversarial/mplight.yml
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import yaml


# ---------------------------------------------------------------------------
# YAML config loader
# ---------------------------------------------------------------------------

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Reward function patches
# ---------------------------------------------------------------------------

class OldRewardMixin:
    """Original reward: penalises absolute delay."""
    def calculate_reward(self, real_delay, total_fake_vehicles):
        reward = -real_delay - self.penalty_lambda * total_fake_vehicles
        return reward


class NewRewardMixin:
    """Delta-delay reward: rewards actively increasing delay each step."""
    def calculate_reward(self, real_delay, total_fake_vehicles):
        if self.prev_delay is None:
            delta_delay = 0.0
        else:
            delta_delay = real_delay - self.prev_delay
        self.prev_delay = real_delay
        reward = delta_delay - self.penalty_lambda * total_fake_vehicles
        return reward


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_experiment(trainer, reward_label, cfg, seed=42):
    """
    Run one full training experiment and return per-episode travel times.

    Args:
        trainer:       Instantiated TSCTrainerAttacker
        reward_label:  'old' or 'new'
        cfg:           Parsed YAML config dict
        seed:          Random seed

    Returns:
        List of mean average travel times, one per episode
    """
    np.random.seed(seed)
    steps = 3600
    episodes        = cfg['trainer']['episodes']
    steps           = cfg['trainer']['steps']
    action_interval = cfg['trainer']['action_interval']

    travel_times = []

    for e in range(episodes):
        trainer.world.reset()

        for ag in trainer.agents:
            ag.reset()
            # Patch the reward function on the injector at runtime
            if reward_label == 'old':
                ag.injector.__class__.calculate_reward = OldRewardMixin.calculate_reward
                ag.injector.prev_delay = None
            else:
                ag.injector.__class__.calculate_reward = NewRewardMixin.calculate_reward
                ag.injector.prev_delay = None

        episode_travel_times = []
        i = 0

        while i < steps:
            approach_actions, scale_actions = [], []
            for ag in trainer.agents:
                state = ag.get_state()
                action, _ = ag.get_action(state, test=False)
                approach_actions.append(action[0])
                scale_actions.append(action[1])

            trainer.world.reset_fake_vehicles()

            total_fake = 0
            for idx, ag in enumerate(trainer.agents):
                approach = approach_actions[idx]
                scales   = scale_actions[idx]
                total_fake += ag.injector.inject_approach_vehicles(
                    ag._approaches[approach % len(ag._approaches)], scales
                )

            actions = np.random.randint(0, 4, len(trainer.agents))

            for _ in range(action_interval):
                trainer.world.step(actions)
                trainer.world.update_current_measurements()
                i += 1

                real_delay = trainer.world.eng.get_average_travel_time()

                # Call whichever reward version is patched in
                _ = trainer.agents[0].injector.calculate_reward(real_delay, total_fake)

                episode_travel_times.append(real_delay)

                # Policy update
                update_rate = cfg['trainer'].get('update_model_rate', 1)
                if i % update_rate == 0:
                    for ag in trainer.agents:
                        ag.update_policy()

                if i >= steps:
                    break

        mean_tt = float(np.mean(episode_travel_times)) if episode_travel_times else 0.0
        travel_times.append(mean_tt)
        print(f"[{reward_label.upper():3s}] Episode {e+1:>3}/{episodes} — "
              f"Mean Travel Time: {mean_tt:.2f}s")

    return travel_times


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_comparison(old_tt, new_tt, save_path="reward_comparison.png"):
    episodes = range(1, len(old_tt) + 1)

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(episodes, old_tt,
            label="Old  (−delay − λ·fake)",
            color="#185FA5", linewidth=2, marker='o', markersize=4)
    ax.plot(episodes, new_tt,
            label="New  (Δdelay − λ·fake)",
            color="#D85A30", linewidth=2, marker='s', markersize=4)

    ax.fill_between(episodes, old_tt, new_tt,
                    where=[n > o for n, o in zip(new_tt, old_tt)],
                    alpha=0.12, color="#D85A30", label="New > Old")
    ax.fill_between(episodes, old_tt, new_tt,
                    where=[o >= n for n, o in zip(new_tt, old_tt)],
                    alpha=0.12, color="#185FA5", label="Old > New")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Mean Average Travel Time (s)", fontsize=12)
    ax.set_title("Old vs New Reward — Average Travel Time per Episode", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"\nPlot saved → {save_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(old_tt, new_tt):
    print("\n" + "="*54)
    print(f"{'Metric':<32} {'Old':>10} {'New':>10}")
    print("-"*54)
    print(f"{'Mean travel time (s)':<32} {np.mean(old_tt):>10.2f} {np.mean(new_tt):>10.2f}")
    print(f"{'Max travel time (s)':<32} {np.max(old_tt):>10.2f} {np.max(new_tt):>10.2f}")
    print(f"{'Min travel time (s)':<32} {np.min(old_tt):>10.2f} {np.min(new_tt):>10.2f}")
    print(f"{'Std dev (s)':<32} {np.std(old_tt):>10.2f} {np.std(new_tt):>10.2f}")
    delta   = np.mean(new_tt) - np.mean(old_tt)
    pct     = (delta / np.mean(old_tt)) * 100 if np.mean(old_tt) != 0 else 0
    verdict = "NEW reward is MORE disruptive" if delta > 0 else "OLD reward is MORE disruptive"
    print(f"\n  Delta (new - old): {delta:+.2f}s ({pct:+.1f}%)")
    print(f"  Verdict: {verdict}")
    print("="*54 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="e.g. configs/tsc_rl_adversarial/mplight.yml")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--output", type=str, default="reward_comparison.png")
    args = parser.parse_args()

    # Load YAML
    cfg = load_config(args.config)
    print(f"Loaded config : {args.config}")
    print(f"  episodes       = {cfg['trainer']['episodes']}")
    print(f"  steps          = {cfg['trainer']['steps']}")
    print(f"  action_interval= {cfg['trainer']['action_interval']}")
    print(f"  penalty_lambda = {cfg['attacker']['penalty_lambda']}")

    # Bootstrap registry and trainer the same way your main.py does
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common.registry import Registry
    from attacker.trainer import TSCTrainerAttacker

    Registry.mapping['command_mapping']['setting'].param.update({
        'config':      args.config,
        'world':       cfg['world'].get('engine', 'cityflow'),
        'thread_num':  cfg['trainer'].get('thread', 4),
        'interface':   cfg['world'].get('interface', 'cityflow'),
        'agent':       cfg['model'].get('name', 'tsc_attacker'),
        'delay_type':  'apx',
    })

    trainer = TSCTrainerAttacker(logger=None)
    trainer.create_world()
    trainer.create_agents()

    print("\n--- Running OLD reward ---")
    old_tt = run_experiment(trainer, 'old', cfg, seed=args.seed)

    print("\n--- Running NEW reward (delta delay) ---")
    new_tt = run_experiment(trainer, 'new', cfg, seed=args.seed)

    print_summary(old_tt, new_tt)
    plot_comparison(old_tt, new_tt, save_path=args.output)
