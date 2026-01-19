#!/usr/bin/env python3
import os
import csv
import argparse
import matplotlib.pyplot as plt

def read_delay_csv(path: str):
    steps, delays = [], []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        # expects columns: step,delay
        for row in reader:
            try:
                steps.append(int(float(row["step"])))
                delays.append(float(row["delay"]))
            except Exception:
                # skip malformed rows
                continue
    return steps, delays

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="jinan", help="scenario name under output_data/baselines/")
    ap.add_argument("--out", default=None, help="output png path (optional)")
    ap.add_argument("--show", action="store_true", help="show plot window")
    args = ap.parse_args()

    base_dir = os.path.join("output_data", "baselines", args.scenario)

    agents = ["maxpressure", "fixedtime", "random", "mplight"]
    found_any = False

    plt.figure()
    for agent in agents:
        csv_path = os.path.join(base_dir, agent, "delay.csv")
        if not os.path.exists(csv_path):
            print(f"[skip] missing: {csv_path}")
            continue
        steps, delays = read_delay_csv(csv_path)
        if not steps:
            print(f"[skip] empty/unreadable: {csv_path}")
            continue
        found_any = True
        plt.plot(steps, delays, label=agent)

    if not found_any:
        raise SystemExit(f"No delay.csv files found under {base_dir}/<agent>/delay.csv")

    plt.xlabel("Step")
    plt.ylabel("Cumulative delay (seconds)")
    plt.title(f"Delay curves — {args.scenario}")
    plt.legend()
    plt.tight_layout()

    out_path = args.out or os.path.join(base_dir, "delay_curves.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"[saved] {out_path}")

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
