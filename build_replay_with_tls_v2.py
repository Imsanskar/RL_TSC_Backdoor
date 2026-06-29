"""
Convert SUMO FCD XML + per-step TLS log to CityFlow replay format
with both vehicles and signal lights animating.

Inputs:
    --fcd       SUMO FCD XML (from --fcd-output)
    --tls       Per-step TLS log (from TraCI in world_sumo.py)
    --netxml    SUMO net.xml (to map linkIndex -> from_road, from_lane)
    --roadnet   CityFlow roadnet.json (for road geometry and lane counts)
    --output    Output replay .txt

The CityFlow replay format per timestep:
    <vehicle1>,<vehicle2>,...;<road_id> <lane_states>,<road_id> <lane_states>,...

Each vehicle:  x y angle vehicle_id 0 length width
Each road TLS: road_id <state per CityFlow lane separated by spaces>

Where lane state is mapped from SUMO link letters:
    G/g/o = green        (CityFlow uses 'g')
    y     = yellow       (CityFlow uses 'y')
    r/R   = red          (CityFlow uses 'r')

We aggregate connection states per (from_road, from_lane). If a lane has any
green connection it's marked green; else yellow if any yellow; else red.
"""

import xml.etree.ElementTree as ET
import json
import math
import argparse


def load_roadnet(roadnet_path):
    """Load CityFlow roadnet -> dict of road_id -> geometry."""
    with open(roadnet_path) as f:
        data = json.load(f)
    if 'static' in data:
        data = data['static']
    if 'roads' in data:
        edge_list = data['roads']
    elif 'edges' in data:
        edge_list = data['edges']
    else:
        edge_list = data.get('roadnetwork', {}).get('edges', [])

    roads = {}
    for edge in edge_list:
        road_id = edge['id']
        # Points may be:
        #   - List of dicts: [{"x":..,"y":..}, ...]    (CityFlow source roadnet.json)
        #   - List of pairs: [[x,y], [x,y]]            (CityFlow log dump)
        if 'points' not in edge:
            continue
        points = edge['points']
        def _xy(pt):
            if isinstance(pt, dict):
                return float(pt['x']), float(pt['y'])
            return float(pt[0]), float(pt[1])
        p_start = _xy(points[0])
        p_end = _xy(points[-1])

        dx = p_end[0] - p_start[0]
        dy = p_end[1] - p_start[1]
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        ux, uy = dx / length, dy / length
        rx, ry = uy, -ux

        # Lane widths come from one of:
        #   - 'laneWidths': [4.0, 4.0, ...]                  (log dump)
        #   - 'lanes': [{"width": 4.0, ...}, ...]            (CityFlow source)
        if 'laneWidths' in edge:
            lane_widths = edge['laneWidths']
            n_lanes = len(lane_widths)
        elif 'lanes' in edge and isinstance(edge['lanes'], list):
            lane_widths = [float(l.get('width', 4.0)) for l in edge['lanes']]
            n_lanes = len(lane_widths)
        else:
            n_lanes = edge.get('nLane', 2) or 2
            lane_widths = [4.0] * n_lanes

        lane_offsets = []
        cumulative = 0.0
        for w in lane_widths:
            lane_offsets.append(cumulative + w / 2.0)
            cumulative += w

        angle_rad = math.atan2(uy, ux)
        if angle_rad < 0:
            angle_rad += 2 * math.pi

        roads[road_id] = {
            'start': p_start,
            'unit': (ux, uy),
            'right': (rx, ry),
            'lane_offsets': lane_offsets,
            'n_lanes': n_lanes,
            'angle': angle_rad,
        }
    return roads


def load_link_mapping(netxml_path):
    """
    From SUMO net.xml, build:
       link_map[tls_id][link_index] = (from_road, from_lane)
    """
    tree = ET.parse(netxml_path)
    root = tree.getroot()

    link_map = {}
    for conn in root.findall('connection'):
        tl = conn.get('tl')
        if not tl:
            continue
        try:
            li = int(conn.get('linkIndex'))
        except (TypeError, ValueError):
            continue
        from_road = conn.get('from')
        from_lane = int(conn.get('fromLane', '0'))
        if tl not in link_map:
            link_map[tl] = {}
        link_map[tl][li] = (from_road, from_lane)
    return link_map


def sumo_char_to_cf(c):
    """Map SUMO link letter -> CityFlow signal letter."""
    if c in ('G', 'g', 'o'):
        return 'g'
    if c in ('y', 'Y'):
        return 'y'
    return 'r'   # 'r', 'R', 's', and anything else default to red


def state_priority(c):
    """For aggregating multiple connection states on one lane: g > y > r."""
    return {'g': 2, 'y': 1, 'r': 0}.get(c, 0)


def load_tls_per_step(tls_path):
    """
    Parse the per-step TLS log file.
    Returns: dict[int_step] -> dict[tls_id] -> state_string
    """
    per_step = {}
    with open(tls_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                t = float(parts[0])
            except ValueError:
                continue
            step = int(round(t))
            tls_states = {}
            for token in parts[1:]:
                if '=' in token:
                    tid, state = token.split('=', 1)
                    tls_states[tid] = state
            per_step[step] = tls_states
    return per_step


def build_tls_str_for_step(tls_states, link_map, roads, incoming_roads):
    """
    From the SUMO state string at this step, build the CityFlow TLS section.
    Format: "road_id g r,road_id r g,..." (one entry per INCOMING road)
    """
    # lane_state[road_id][lane_idx] = best char so far
    lane_state = {}

    for tls_id, state_str in tls_states.items():
        if tls_id not in link_map:
            continue
        for li, ch in enumerate(state_str):
            if li not in link_map[tls_id]:
                continue
            from_road, from_lane = link_map[tls_id][li]
            cf = sumo_char_to_cf(ch)
            if from_road not in lane_state:
                lane_state[from_road] = {}
            prev = lane_state[from_road].get(from_lane, 'r')
            if state_priority(cf) > state_priority(prev):
                lane_state[from_road][from_lane] = cf

    # Now build the per-road string in the order roads appear in the roadnet
    road_entries = []
    # Only include INCOMING roads (those that have at least one connection from them
    # to a signalized intersection). Outgoing roads don't have signals at their start.
    for road_id, road in roads.items():
        if road_id not in incoming_roads:
            continue
        n = road['n_lanes']
        states_for_road = lane_state.get(road_id, {})
        # One char per CityFlow lane.
        # If we have no data for this specific lane → red (conservative).
        # Do NOT fall back to other lanes on the same road — that turns everything green.
        per_lane = []
        for cf_lane_idx in range(n):
            sumo_lane = n - 1 - cf_lane_idx
            ch = states_for_road.get(sumo_lane, 'r')
            per_lane.append(ch)
        road_entries.append(f"{road_id} " + " ".join(per_lane))

    return ",".join(road_entries) + ","


def convert(fcd_path, tls_path, netxml_path, roadnet_path, output_path,
            invert_lane_order=True,
            default_length=5.0, default_width=2.0):
    roads = load_roadnet(roadnet_path)
    link_map = load_link_mapping(netxml_path)
    tls_per_step = load_tls_per_step(tls_path)

    # Identify which roads are INCOMING (have connections from them via the TLS)
    incoming_roads = set()
    for tls_id, mapping in link_map.items():
        for li, (from_road, _) in mapping.items():
            incoming_roads.add(from_road)

    print(f"Loaded {len(roads)} roads")
    print(f"Incoming (signalized) roads: {sorted(incoming_roads)}")
    print(f"Loaded link mapping for {len(link_map)} TLS")
    print(f"Loaded {len(tls_per_step)} TLS timesteps")

    last_tls_states = {}    # carry forward last known state if a step is missing
    last_tls_str = ""
    step_count = 0

    with open(output_path, 'w') as out:
        try:
            for event, elem in ET.iterparse(fcd_path, events=('end',)):
                if elem.tag != 'timestep':
                    continue

                t = float(elem.attrib.get('time', '0'))
                step = int(round(t))

                # Vehicles
                vehicle_records = []
                for v in elem.findall('vehicle'):
                    lane_str = v.attrib.get('lane', '')
                    if lane_str.startswith(':') or not lane_str.startswith('road_'):
                        continue
                    rsplit = lane_str.rsplit('_', 1)
                    if len(rsplit) != 2:
                        continue
                    road_id, lane_idx_str = rsplit
                    try:
                        sumo_lane = int(lane_idx_str)
                    except ValueError:
                        continue
                    if road_id not in roads:
                        continue
                    road = roads[road_id]
                    n = road['n_lanes']
                    if sumo_lane < 0 or sumo_lane >= n:
                        continue
                    cf_lane = n - 1 - sumo_lane if invert_lane_order else sumo_lane
                    pos = float(v.attrib['pos'])
                    ux, uy = road['unit']
                    rx, ry = road['right']
                    px, py = road['start']
                    lo = road['lane_offsets'][cf_lane]
                    x = px + ux * pos + rx * lo
                    y = py + uy * pos + ry * lo
                    angle = road['angle']
                    vid = v.attrib['id']
                    status = "0"
                    vehicle_records.append(
                        f"{x} {y} {angle} {vid} {status} {default_length} {default_width}"
                    )

                # TLS section (carry forward if missing for this step)
                if step in tls_per_step:
                    last_tls_states = tls_per_step[step]
                    last_tls_str = build_tls_str_for_step(last_tls_states, link_map, roads, incoming_roads)
                # else: keep last_tls_str

                # Add trailing comma before ; to match working CityFlow format
                if vehicle_records:
                    line = ",".join(vehicle_records) + ",;" + last_tls_str
                else:
                    line = ";" + last_tls_str
                out.write(line + "\n")
                step_count += 1
                elem.clear()
        except ET.ParseError as e:
            print(f"Warning: parse error: {e}")

    print(f"Wrote {step_count} timesteps to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fcd', required=True, help='SUMO FCD XML')
    parser.add_argument('--tls', required=True, help='Per-step TLS log txt')
    parser.add_argument('--netxml', required=True, help='SUMO net.xml (for linkIndex map)')
    parser.add_argument('--roadnet', required=True, help='CityFlow roadnet.json')
    parser.add_argument('--output', required=True, help='Output replay .txt')
    parser.add_argument('--no-invert-lanes', action='store_true')
    args = parser.parse_args()

    convert(
        fcd_path=args.fcd,
        tls_path=args.tls,
        netxml_path=args.netxml,
        roadnet_path=args.roadnet,
        output_path=args.output,
        invert_lane_order=not args.no_invert_lanes,
    )
