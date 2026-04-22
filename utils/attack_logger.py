"""
Attack analysis logger for adversarial traffic-control experiments.

Writes:
1. step_log.csv for per-decision records
2. episode_summary.jsonl for aggregate episode summaries
"""

import csv
import json
import os
from datetime import datetime

import numpy as np


class AttackAnalysisLogger:
    """Structured logger for attacker/controller interaction analysis."""

    def __init__(self, output_dir, experiment_name=None):
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.output_dir = os.path.join(output_dir, "attack_analysis", experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.step_log_path = os.path.join(self.output_dir, "step_log.csv")
        self.episode_log_path = os.path.join(self.output_dir, "episode_summary.jsonl")
        self.step_fields = [
            "episode",
            "step",
            "attacker_idx",
            "approach_action",
            "approach_name",
            "scale_seg0",
            "scale_seg1",
            "scale_seg2",
            "scale_seg3",
            "total_vehicles_injected",
            "attacker_reward",
            "attacker_value",
            "controller_phase_before",
            "controller_action",
            "queue_per_approach",
            "total_queue",
            "mean_reward",
            "mean_delay",
            "travel_time_so_far",
            "throughput_so_far",
        ]

        with open(self.step_log_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.step_fields)
            writer.writeheader()

        self._reset_episode_accumulators()

    def _reset_episode_accumulators(self):
        self.episode_approach_counts = {}
        self.episode_scale_values = []
        self.episode_rewards = []
        self.episode_vehicles_injected = []
        self.episode_queues = []
        self.episode_controller_actions = []
        self.episode_steps = 0

    def log_step(
        self,
        episode,
        step,
        attacker_idx,
        approach_action,
        scale_action,
        vehicles_injected,
        attacker_reward,
        attacker_value,
        controller_phase_before,
        controller_action,
        queue_lengths,
        mean_reward,
        mean_delay,
        travel_time_so_far=None,
        throughput_so_far=None,
    ):
        approach_names = ["N", "E", "S", "W"]
        safe_approach = int(approach_action) if approach_action is not None else -1
        approach_name = (
            approach_names[safe_approach % len(approach_names)]
            if safe_approach >= 0
            else "NA"
        )

        if isinstance(scale_action, np.ndarray):
            scale_action = scale_action.tolist()
        elif scale_action is None:
            scale_action = []
        elif not isinstance(scale_action, list):
            scale_action = [scale_action]
        scale_padded = (scale_action + [0, 0, 0, 0])[:4]

        if isinstance(queue_lengths, np.ndarray):
            queue_lengths = queue_lengths.tolist()
        elif queue_lengths is None:
            queue_lengths = []
        elif not isinstance(queue_lengths, list):
            queue_lengths = [queue_lengths]
        total_queue = float(sum(queue_lengths)) if queue_lengths else 0.0

        row = {
            "episode": episode,
            "step": step,
            "attacker_idx": attacker_idx,
            "approach_action": safe_approach,
            "approach_name": approach_name,
            "scale_seg0": scale_padded[0],
            "scale_seg1": scale_padded[1],
            "scale_seg2": scale_padded[2],
            "scale_seg3": scale_padded[3],
            "total_vehicles_injected": int(vehicles_injected or 0),
            "attacker_reward": round(float(attacker_reward or 0.0), 6),
            "attacker_value": round(float(attacker_value or 0.0), 6),
            "controller_phase_before": int(controller_phase_before)
            if controller_phase_before is not None
            else -1,
            "controller_action": int(controller_action)
            if controller_action is not None
            else -1,
            "queue_per_approach": json.dumps(queue_lengths),
            "total_queue": round(total_queue, 4),
            "mean_reward": round(float(mean_reward or 0.0), 6),
            "mean_delay": round(float(mean_delay or 0.0), 6),
            "travel_time_so_far": round(float(travel_time_so_far or 0.0), 4),
            "throughput_so_far": int(throughput_so_far or 0),
        }

        with open(self.step_log_path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.step_fields)
            writer.writerow(row)

        if safe_approach >= 0:
            self.episode_approach_counts[safe_approach] = (
                self.episode_approach_counts.get(safe_approach, 0) + 1
            )
        self.episode_scale_values.append(scale_padded)
        self.episode_rewards.append(float(attacker_reward or 0.0))
        self.episode_vehicles_injected.append(int(vehicles_injected or 0))
        self.episode_queues.append(total_queue)
        self.episode_controller_actions.append(
            int(controller_action) if controller_action is not None else -1
        )
        self.episode_steps += 1

    def log_episode_summary(
        self,
        episode,
        travel_time_attack,
        travel_time_no_attack=None,
        total_throughput=None,
        total_delay=None,
        mean_queue=None,
        extra_info=None,
    ):
        total_actions = sum(self.episode_approach_counts.values()) or 1
        approach_dist = {
            ["N", "E", "S", "W"][key]: round(value / total_actions * 100, 1)
            for key, value in sorted(self.episode_approach_counts.items())
        }

        scale_arr = (
            np.array(self.episode_scale_values, dtype=float)
            if self.episode_scale_values
            else np.zeros((1, 4), dtype=float)
        )
        tt_increase_pct = None
        if travel_time_no_attack not in (None, 0):
            tt_increase_pct = round(
                (float(travel_time_attack) - float(travel_time_no_attack))
                / float(travel_time_no_attack)
                * 100.0,
                2,
            )

        controller_action_counts = {}
        for action in self.episode_controller_actions:
            controller_action_counts[action] = controller_action_counts.get(action, 0) + 1

        summary = {
            "episode": episode,
            "travel_time_attack": round(float(travel_time_attack or 0.0), 4),
            "travel_time_no_attack": round(float(travel_time_no_attack), 4)
            if travel_time_no_attack not in (None, 0)
            else None,
            "travel_time_increase_pct": tt_increase_pct,
            "total_throughput": int(total_throughput) if total_throughput is not None else None,
            "total_delay": round(float(total_delay), 4) if total_delay is not None else None,
            "mean_queue": round(float(mean_queue), 4) if mean_queue is not None else None,
            "approach_distribution_pct": approach_dist,
            "scale_mean_per_segment": scale_arr.mean(axis=0).round(2).tolist(),
            "scale_std_per_segment": scale_arr.std(axis=0).round(2).tolist(),
            "mean_vehicles_injected_per_step": round(
                float(np.mean(self.episode_vehicles_injected)) if self.episode_vehicles_injected else 0.0,
                2,
            ),
            "total_vehicles_injected": int(sum(self.episode_vehicles_injected)),
            "mean_attacker_reward": round(
                float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
                6,
            ),
            "std_attacker_reward": round(
                float(np.std(self.episode_rewards)) if self.episode_rewards else 0.0,
                6,
            ),
            "total_attacker_reward": round(float(sum(self.episode_rewards)), 4),
            "mean_queue_during_attack": round(
                float(np.mean(self.episode_queues)) if self.episode_queues else 0.0,
                4,
            ),
            "max_queue_during_attack": round(
                float(max(self.episode_queues)) if self.episode_queues else 0.0,
                4,
            ),
            "controller_action_distribution": controller_action_counts,
            "total_steps": self.episode_steps,
        }

        if extra_info:
            summary.update(extra_info)

        with open(self.episode_log_path, "a") as handle:
            handle.write(json.dumps(summary) + "\n")

        self._reset_episode_accumulators()

    def get_output_dir(self):
        return self.output_dir
