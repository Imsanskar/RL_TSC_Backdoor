#!/usr/bin/env python3
"""
Compare two attack-analysis directories produced by AttackAnalysisLogger.

Example:
python3 utils/analyze_attack.py \
    --dir1 data/output_data/.../attack_analysis/TEST_cityflow1x1 \
    --name1 "Baseline 1x1" \
    --dir2 data/output_data/.../attack_analysis/TEST_cityflow1x1_sb_sx_attacker_from_cityflow1x1 \
    --name2 "Zero-Shot 1x1->1x1_sb_sx"
"""

import argparse
import json
import os
from typing import Dict

import pandas as pd


def load_experiment(exp_dir: str):
    step_path = os.path.join(exp_dir, "step_log.csv")
    episode_path = os.path.join(exp_dir, "episode_summary.jsonl")

    if not os.path.exists(step_path):
        raise FileNotFoundError(f"Missing step log: {step_path}")
    if not os.path.exists(episode_path):
        raise FileNotFoundError(f"Missing episode summary: {episode_path}")

    step_df = pd.read_csv(step_path)
    summaries = []
    with open(episode_path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                summaries.append(json.loads(line))
    summary_df = pd.DataFrame(summaries)
    return step_df, summary_df


def normalized_counts(series: pd.Series) -> Dict[str, float]:
    if len(series) == 0:
        return {}
    counts = series.value_counts(normalize=True).sort_index()
    return {str(k): round(float(v), 4) for k, v in counts.items()}


def describe_run(name: str, step_df: pd.DataFrame, summary_df: pd.DataFrame):
    print(f"\n=== {name} ===")
    print(f"Rows in step log: {len(step_df)}")

    if len(summary_df) > 0:
        latest = summary_df.iloc[-1].to_dict()
        print("Latest episode summary:")
        for key in [
            "travel_time_attack",
            "travel_time_increase_pct",
            "mean_attacker_reward",
            "mean_queue",
            "mean_queue_during_attack",
            "mean_vehicles_injected_per_step",
            "total_vehicles_injected",
            "target_network",
            "attacker_source",
            "controller_source",
            "mode",
        ]:
            if key in latest and pd.notna(latest[key]):
                print(f"  {key}: {latest[key]}")

    print("Approach distribution:")
    print(normalized_counts(step_df["approach_name"]))

    scale_cols = [c for c in ["scale_seg0", "scale_seg1", "scale_seg2", "scale_seg3"] if c in step_df.columns]
    if scale_cols:
        print("Mean scale per segment:")
        print(step_df[scale_cols].mean().round(4).to_dict())

    if "controller_action" in step_df.columns:
        print("Controller action distribution:")
        print(normalized_counts(step_df["controller_action"]))

    for col in [
        "total_vehicles_injected",
        "attacker_reward",
        "mean_delay",
        "travel_time_so_far",
        "throughput_so_far",
        "total_queue",
    ]:
        if col in step_df.columns:
            print(f"{col}: mean={step_df[col].mean():.4f}, std={step_df[col].std():.4f}")


def compare_runs(name1: str, df1: pd.DataFrame, name2: str, df2: pd.DataFrame):
    print(f"\n=== Comparison: {name1} vs {name2} ===")
    scale_cols = [c for c in ["scale_seg0", "scale_seg1", "scale_seg2", "scale_seg3"] if c in df1.columns and c in df2.columns]
    if scale_cols:
        diff = (df2[scale_cols].mean() - df1[scale_cols].mean()).round(4).to_dict()
        print("Delta mean scale per segment (run2 - run1):")
        print(diff)

    if "approach_name" in df1.columns and "approach_name" in df2.columns:
        print("Approach distribution delta:")
        keys = sorted(set(df1["approach_name"].dropna().unique()) | set(df2["approach_name"].dropna().unique()))
        dist1 = df1["approach_name"].value_counts(normalize=True)
        dist2 = df2["approach_name"].value_counts(normalize=True)
        delta = {str(k): round(float(dist2.get(k, 0.0) - dist1.get(k, 0.0)), 4) for k in keys}
        print(delta)

    if "controller_action" in df1.columns and "controller_action" in df2.columns:
        print("Controller action distribution delta:")
        keys = sorted(set(df1["controller_action"].dropna().unique()) | set(df2["controller_action"].dropna().unique()))
        dist1 = df1["controller_action"].value_counts(normalize=True)
        dist2 = df2["controller_action"].value_counts(normalize=True)
        delta = {str(k): round(float(dist2.get(k, 0.0) - dist1.get(k, 0.0)), 4) for k in keys}
        print(delta)

    for col in ["total_vehicles_injected", "attacker_reward", "mean_delay", "travel_time_so_far", "throughput_so_far", "total_queue"]:
        if col in df1.columns and col in df2.columns:
            print(f"Delta {col} mean (run2 - run1): {(df2[col].mean() - df1[col].mean()):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Compare two attack analysis runs")
    parser.add_argument("--dir1", required=True, help="First attack_analysis experiment directory")
    parser.add_argument("--name1", required=True, help="Display name for first run")
    parser.add_argument("--dir2", required=True, help="Second attack_analysis experiment directory")
    parser.add_argument("--name2", required=True, help="Display name for second run")
    args = parser.parse_args()

    step_df1, summary_df1 = load_experiment(args.dir1)
    step_df2, summary_df2 = load_experiment(args.dir2)

    describe_run(args.name1, step_df1, summary_df1)
    describe_run(args.name2, step_df2, summary_df2)
    compare_runs(args.name1, step_df1, args.name2, step_df2)


if __name__ == "__main__":
    main()
