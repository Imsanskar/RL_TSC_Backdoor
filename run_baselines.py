import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def newest_dir(path):
    candidates = [p for p in Path(path).iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def copy_replay_artifacts(replay_root, baseline_dir):
    baseline_replay_dir = Path(baseline_dir) / "replay"
    baseline_replay_dir.mkdir(parents=True, exist_ok=True)

    for name in ("replay.txt", "replay_roadnet.json", "config_replay.json"):
        src = Path(replay_root) / name
        if src.exists():
            shutil.copy2(src, baseline_replay_dir / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="jinan")
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["maxpressure", "fixedtime", "random", "mplight"],
    )
    args = parser.parse_args()

    for agent in args.agents:
        cmd = [
            sys.executable,
            "test_run.py",
            "--agent",
            agent,
            "--train_scenario",
            args.scenario,
            "--test_scenario",
            args.scenario,
            "--seed",
            str(args.seed),
        ]
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd)

        replay_root = Path("output_data") / "replays" / args.scenario / agent / f"seed_{args.seed}"
        latest = newest_dir(replay_root) if replay_root.exists() else None
        if latest is None:
            print(f"[WARN] No replay found for {agent} in {replay_root}")
            continue

        baseline_dir = Path("output_data") / "baselines" / args.scenario / agent
        copy_replay_artifacts(latest, baseline_dir)
        print(f"Copied replay artifacts to {baseline_dir / 'replay'}")


if __name__ == "__main__":
    main()
