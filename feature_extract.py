"""
Extract features from CityFlow-format traffic signal control datasets.

Each dataset folder contains:
  - config.json   : CityFlow simulator config, points to the roadnet + flow files
  - roadnet file  : road network topology (intersections, roads, lanes)
  - flow file     : vehicle trajectories (routes + departure times)

Usage:
    python extract_features.py data/raw_road_net_data/jinan_2000/
    python extract_features.py data/raw_road_net_data/           # all subfolders
"""

import json
import sys
from pathlib import Path
from collections import Counter


# ---------- FILE RESOLUTION ----------

def resolve_files(folder):
    """
    Figure out which file is the roadnet and which is the flow.
    Strategy:
      1) Prefer config.json -- CityFlow configs declare both filenames explicitly.
      2) Fall back to filename heuristics if config missing or incomplete.
    Returns (roadnet_path, flow_path, config_dict_or_None).
    """
    folder = Path(folder)
    config_path = folder / "config.json"

    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        roadnet_name = cfg.get("roadnetFile") or cfg.get("roadnet") or ""
        flow_name = cfg.get("flowFile") or cfg.get("flow") or ""
        roadnet_path = folder / Path(roadnet_name).name if roadnet_name else None
        flow_path = folder / Path(flow_name).name if flow_name else None
        if roadnet_path and roadnet_path.exists() and flow_path and flow_path.exists():
            return roadnet_path, flow_path, cfg

    # Heuristic fallback: anything with "roadnet" in name is the network,
    # anything else (anon_*, flow_*, traffic_*) is the flow.
    roadnet_path, flow_path = None, None
    for f in folder.iterdir():
        if not (f.is_file() and f.suffix == ".json"):
            continue
        name = f.name.lower()
        if name == "config.json":
            continue
        if "roadnet" in name:
            roadnet_path = f
        else:
            flow_path = f
    return roadnet_path, flow_path, None


# ---------- ROADNET FEATURES ----------

def roadnet_features(roadnet):
    intersections = roadnet.get("intersections", [])
    roads = roadnet.get("roads", [])

    signalized = [i for i in intersections if not i.get("virtual", False)]
    virtual = [i for i in intersections if i.get("virtual", False)]

    lanes_per_road = [len(r.get("lanes", [])) for r in roads]

    # Road lengths (Euclidean from start/end points)
    road_lengths = []
    for r in roads:
        pts = r.get("points", [])
        if len(pts) >= 2:
            x1, y1 = pts[0]["x"], pts[0]["y"]
            x2, y2 = pts[-1]["x"], pts[-1]["y"]
            road_lengths.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)

    # Phase count per signalized intersection (skip all-red / yellow transitions)
    phase_counts = []
    for i in signalized:
        tl = i.get("trafficLight", {})
        phases = tl.get("lightphases", [])
        active = [p for p in phases if p.get("availableRoadLinks")]
        phase_counts.append(len(active))

    # Rough grid inference from signalized intersection coordinates
    xs = [i["point"]["x"] for i in signalized if "point" in i]
    ys = [i["point"]["y"] for i in signalized if "point" in i]
    unique_x = len({round(x, -1) for x in xs})  # bucket by 10m to absorb jitter
    unique_y = len({round(y, -1) for y in ys})

    return {
        "n_intersections_total": len(intersections),
        "n_intersections_signalized": len(signalized),
        "n_intersections_virtual": len(virtual),
        "approx_grid": f"{unique_x} x {unique_y}",
        "n_roads": len(roads),
        "lanes_per_road_avg": round(sum(lanes_per_road) / len(lanes_per_road), 2) if lanes_per_road else 0,
        "lanes_per_road_range": f"{min(lanes_per_road)}-{max(lanes_per_road)}" if lanes_per_road else "n/a",
        "road_length_avg_m": round(sum(road_lengths) / len(road_lengths), 1) if road_lengths else 0,
        "road_length_range_m": f"{min(road_lengths):.0f}-{max(road_lengths):.0f}" if road_lengths else "n/a",
        "phases_per_signal_avg": round(sum(phase_counts) / len(phase_counts), 2) if phase_counts else 0,
    }


# ---------- FLOW FEATURES ----------

def flow_features(flow):
    if not flow:
        return {"n_vehicles": 0}

    n_vehicles = len(flow)
    start_times = [v.get("startTime", 0) for v in flow]
    end_times = [v.get("endTime", v.get("startTime", 0)) for v in flow]
    route_lens = [len(v.get("route", [])) for v in flow]

    # Vehicle physical params (first entry; usually uniform across the file)
    vp = flow[0].get("vehicle", {})

    # Arrival rate: vehicles per 5-minute window (standard TSC metric)
    span = max(end_times) - min(start_times) if n_vehicles else 1
    span = max(span, 1)
    per_5min = n_vehicles / span * 300

    # OD diversity: unique (first road, last road) pairs
    od_pairs = Counter()
    for v in flow:
        route = v.get("route", [])
        if route:
            od_pairs[(route[0], route[-1])] += 1

    return {
        "n_vehicles": n_vehicles,
        "time_window_s": f"{min(start_times):.0f}-{max(start_times):.0f}",
        "arrival_rate_per_5min": round(per_5min, 1),
        "route_length_avg_roads": round(sum(route_lens) / len(route_lens), 2) if route_lens else 0,
        "route_length_range": f"{min(route_lens)}-{max(route_lens)}" if route_lens else "n/a",
        "unique_OD_pairs": len(od_pairs),
        "vehicle_max_speed": vp.get("maxSpeed", "n/a"),
        "vehicle_length": vp.get("length", "n/a"),
        "vehicle_min_gap": vp.get("minGap", "n/a"),
    }


# ---------- CONFIG FEATURES ----------

def config_features(cfg):
    """Simulator-level info from CityFlow config.json."""
    if not cfg:
        return {}
    return {
        "interval_s": cfg.get("interval", "n/a"),
        "seed": cfg.get("seed", "n/a"),
        "rlTrafficLight": cfg.get("rlTrafficLight", "n/a"),
        "laneChange": cfg.get("laneChange", "n/a"),
    }


# ---------- DRIVER ----------

def analyze_dataset(folder):
    folder = Path(folder)
    roadnet_path, flow_path, cfg = resolve_files(folder)

    result = {"dataset": folder.name, "_files": {}}

    if cfg:
        result["config"] = config_features(cfg)

    if roadnet_path and roadnet_path.exists():
        result["_files"]["roadnet"] = roadnet_path.name
        with open(roadnet_path) as f:
            result["roadnet"] = roadnet_features(json.load(f))
    else:
        result["_files"]["roadnet"] = "NOT FOUND"

    if flow_path and flow_path.exists():
        result["_files"]["flow"] = flow_path.name
        with open(flow_path) as f:
            result["flow"] = flow_features(json.load(f))
    else:
        result["_files"]["flow"] = "NOT FOUND"

    return result


def pretty_print(result):
    print(f"\n=== {result['dataset']} ===")
    files = result.get("_files", {})
    if files:
        print(f"  files: roadnet={files.get('roadnet')}, flow={files.get('flow')}")
    for section in ("config", "roadnet", "flow"):
        if section not in result:
            continue
        print(f"  [{section}]")
        for k, v in result[section].items():
            print(f"    {k:30s} {v}")


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw_road_net_data/")
    if not target.exists():
        print(f"Path not found: {target}")
        sys.exit(1)

    # Single dataset if it directly contains JSON files; otherwise iterate subfolders.
    has_json_here = any(f.suffix == ".json" for f in target.iterdir() if f.is_file())
    datasets = [target] if has_json_here else sorted(p for p in target.iterdir() if p.is_dir())

    for d in datasets:
        try:
            pretty_print(analyze_dataset(d))
        except Exception as e:
            print(f"\n=== {d.name} ===  (error: {e})")
