"""
SDSM Injector - Generates and injects fake vehicles into the simulation.

This module creates fake vehicles that appear in the TSC controller's perception
data to manipulate traffic signal decisions.
"""

import numpy as np
from common.registry import Registry


@Registry.register_model('sdsm_injector')
class SDSMInjector:
    """
    Injects fake vehicles into the CityFlow simulation via SDSM forgery.

    The attacker can:
    1. Select which approach (N/E/S/W) to target
    2. Inject varying numbers of fake vehicles per segment
    3. Set fake vehicle speeds (stationary or slow-moving)
    """

    def __init__(self, world, intersection_id, max_vehicles_per_segment=10,
                 num_segments=3, penalty_lambda=0.01):
        """
        Initialize SDSM injector.

        Args:
            world: World object with CityFlow engine
            intersection_id: ID of the intersection to attack
            max_vehicles_per_segment: Maximum fake vehicles per segment (0-10)
            num_segments: Number of segments upstream per approach (default: 3)
            penalty_lambda: Penalty coefficient for fake vehicle count in reward
        """
        self.world = world
        self.intersection_id = intersection_id
        self.max_vehicles_per_segment = max_vehicles_per_segment
        self.num_segments = num_segments
        self.penalty_lambda = penalty_lambda

        # Get intersection object
        self.intersection = world.id2intersection[intersection_id]

        # Track injected fake vehicles for cleanup
        self.injected_vehicles = set()

        # Fake vehicle ID counter
        self.fake_vehicle_counter = 0

        # Current injection plan (approach, segment_counts)
        self.current_plan = None

        # Number of incoming roads (approaches)
        self.num_approaches = len(self.intersection.in_roads)
        self.approaches = ['N', 'E', 'S', 'W']  # Clockwise order

        # Map approach to road index
        self.approach_to_idx = {app: i for i, app in enumerate(self.approaches)}

    def reset(self):
        """Reset injector state."""
        self.injected_vehicles = set()
        self.fake_vehicle_counter = 0
        self.current_plan = None

    def get_fake_vehicle_id(self):
        """Generate unique fake vehicle ID."""
        vid = f"fake_{self.intersection_id}_{self.fake_vehicle_counter}"
        self.fake_vehicle_counter += 1
        return vid

    def get_approach_lanes(self, approach):
        """
        Get incoming lanes for a specific approach.

        Args:
            approach: 'N', 'E', 'S', or 'W'

        Returns:
            List of lane IDs for incoming lanes
        """
        idx = self.approach_to_idx.get(approach)
        if idx is None or idx >= len(self.intersection.in_roads):
            return []

        road = self.intersection.in_roads[idx]
        lanes = []
        for i in range(len(road['lanes'])):
            lane_id = road['id'] + "_" + str(i)
            lanes.append(lane_id)
        return lanes

    def calculate_segment_positions(self, road, num_segments=3):
        """
        Calculate positions for segments upstream of stop bar.

        Args:
            road: Road object from roadnet
            num_segments: Number of segments to divide

        Returns:
            List of positions (distances upstream in meters)
        """
        # Get road length
        point_x = road['points'][0]['x'] - road['points'][1]['x']
        point_y = road['points'][0]['y'] - road['points'][1]['y']
        road_length = np.sqrt(point_x**2 + point_y**2)

        # Divide into segments (farthest to nearest)
        positions = []
        for i in range(num_segments):
            # Position from stop bar (farthest segment first)
            pos = road_length * (num_segments - i) / (num_segments + 1)
            positions.append(pos)

        return positions

    def get_stop_bar_position(self, road, approach):
        """
        Get stop bar position for an approach.

        Args:
            road: Road object
            approach: 'N', 'E', 'S', or 'W'

        Returns:
            (x, y) coordinates of stop bar
        """
        if approach == 'N':
            # Coming from north, stop bar is at southern point
            return road['points'][1]['x'], road['points'][1]['y']
        elif approach == 'S':
            return road['points'][0]['x'], road['points'][0]['y']
        elif approach == 'E':
            return road['points'][0]['x'], road['points'][0]['y']
        elif approach == 'W':
            return road['points'][1]['x'], road['points'][1]['y']
        return 0, 0

    def inject_approach_vehicles(self, approach, segment_counts):
        """
        Inject fake vehicles for a specific approach.

        Args:
            approach: 'N', 'E', 'S', or 'W'
            segment_counts: List of vehicle counts per segment

        Returns:
            Number of vehicles injected
        """
        idx = self.approach_to_idx.get(approach)
        if idx is None or idx >= len(self.intersection.in_roads):
            return 0

        road = self.intersection.in_roads[idx]
        lanes = self.get_approach_lanes(approach)
        segment_positions = self.calculate_segment_positions(road, self.num_segments)

        vehicles_injected = 0

        for seg_idx, count in enumerate(segment_counts):
            if seg_idx >= len(segment_positions) or seg_idx >= len(lanes):
                break

            # Use lane 0 (left-turn or straight) for injection
            lane = lanes[0] if lanes else None
            if lane is None:
                continue

            position = segment_positions[seg_idx]
            stop_x, stop_y = self.get_stop_bar_position(road, approach)

            # Calculate position based on approach direction
            if approach == 'N':
                x, y = stop_x, stop_y + position
                direction = 270  # Facing south
            elif approach == 'S':
                x, y = stop_x, stop_y - position
                direction = 90   # Facing north
            elif approach == 'E':
                x, y = stop_x - position, stop_y
                direction = 180  # Facing west
            elif approach == 'W':
                x, y = stop_x + position, stop_y
                direction = 0    # Facing east
            else:
                x, y = stop_x, stop_y
                direction = 0

            # Inject vehicles with different speeds based on segment
            # Closer segments = stationary (queue), farther = slow-moving
            if seg_idx == len(segment_positions) - 1:  # Closest segment
                speed = 0.0  # Stationary (queue)
            else:
                speed = 5.0  # Slow-moving (platoon)

            for _ in range(count):
                vid = self.get_fake_vehicle_id()
                try:
                    # Add vehicle to CityFlow
                    self.world.eng.add_vehicle({
                        'id': vid,
                        'route': self._get_fake_route(approach),
                        'type': 'standard',
                        'lane': lane,
                        ' departPos': str(position),
                        'departSpeed': str(speed),
                        'departAngle': str(direction)
                    })
                    self.injected_vehicles.add(vid)
                    vehicles_injected += 1
                except Exception as e:
                    # Vehicle already exists or other error
                    pass

        return vehicles_injected

    def _get_fake_route(self, approach):
        """Get a valid route for fake vehicles."""
        # Create a simple route that doesn't interfere with real traffic
        return f"fake_route_{approach}"

    def get_state(self):
        """
        Get current state for attacker policy.

        Returns:
            State vector containing:
            - Vehicle counts per segment (per approach)
            - Average speed per segment
            - Current phase (one-hot)
            - Phase duration ratio
        """
        state = []

        # Get vehicle counts per lane
        lane_counts = self.world.eng.get_lane_vehicle_count()
        lane_speeds = self.world.eng.get_vehicle_speed()

        for approach in self.approaches:
            lanes = self.get_approach_lanes(approach)

            for seg_idx in range(self.num_segments):
                # Count vehicles in this segment
                count = 0
                total_speed = 0
                speed_count = 0

                for lane in lanes:
                    vehicles = lane_counts.get(lane, [])
                    count += len(vehicles)

                    for v in vehicles:
                        speed = lane_speeds.get(v, 0)
                        total_speed += speed
                        speed_count += 1

                avg_speed = total_speed / speed_count if speed_count > 0 else 0
                state.append(count)
                state.append(avg_speed)

        # Add current phase (one-hot)
        current_phase = self.intersection.current_phase
        phase_onehot = [0] * len(self.intersection.phases)
        if current_phase < len(phase_onehot):
            phase_onehot[current_phase] = 1
        state.extend(phase_onehot)

        # Add phase duration ratio
        phase_duration = self.intersection.current_phase_time
        state.append(phase_duration)

        return np.array(state, dtype=np.float32)

    def calculate_reward(self, real_delay, total_fake_vehicles):
        """
        Calculate attacker reward.

        Reward = -real_delay - lambda * num_fake_vehicles

        Args:
            real_delay: Total delay to real vehicles
            total_fake_vehicles: Total fake vehicles injected

        Returns:
            Reward value
        """
        reward = -real_delay - self.penalty_lambda * total_fake_vehicles
        return reward

    def cleanup_injected_vehicles(self):
        """Remove all injected fake vehicles from simulation."""
        for vid in list(self.injected_vehicles):
            try:
                self.world.eng.remove_vehicle(vid)
            except:
                pass
        self.injected_vehicles.clear()
