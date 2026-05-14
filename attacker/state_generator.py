"""
Attacker State Generator - Creates observation state for the attacker policy.

This generator provides the attacker with traffic state information including
vehicle counts, speeds, and intersection phase information.
"""

import numpy as np
from common.registry import Registry
from world import world_cityflow, world_sumo

@Registry.register_model('attacker_state_generator')
class AttackerStateGenerator:
    """
    Generate state observations for the attacker policy.

    The state includes:
    - Vehicle counts per segment (per incoming lane)
    - Average speed per segment
    - Current traffic phase (one-hot)
    - Phase duration ratio
    """

    def __init__(self, world, intersection, num_segments=3, num_approaches=4):
        """
        Initialize attacker state generator.

        Args:
            world: World object with CityFlow engine
            intersection: Intersection object to monitor
            num_segments: Number of segments upstream per approach
            num_approaches: Number of approaches (N/E/S/W)
        """
        self.world = world
        self.intersection = intersection
        self.num_segments = num_segments
        self.num_approaches = num_approaches

        # Subscribe to required info functions
        self.world.subscribe(['lane_count', 'lane_waiting_count', 'lane_delay',
                             'time', 'phase'])

        # Calculate state dimension
        # - Vehicle counts: num_approaches * num_segments
        # - Speeds: num_approaches * num_segments
        # - Phase one-hot: len(phases)
        # - Phase duration: 1
        lane_dims = num_approaches * num_segments * 2  # count + speed
        phase_dims = len(intersection.phases) if intersection.phases else 1
        self.ob_length = lane_dims + phase_dims + 1

    def generate(self):
        """
        Generate state observations for the attacker.

        Returns:
            State vector as numpy array
        """
        state = []

        # Get lane data (works for both engines)
        lane_counts, lane_speeds = self._get_lane_data()

        # Get current phase info
        current_phase = self.intersection.current_phase
        phase_duration = self.intersection.current_phase_time

        # Process each approach (N, E, S, W)
        approaches = ['N', 'E', 'S', 'W']
        for approach_idx, approach in enumerate(approaches):
            if approach_idx >= self.num_approaches:
                break

            # Get lanes for this approach
            lanes = self._get_approach_lanes(approach)

            for seg_idx in range(self.num_segments):
                # Count vehicles in this segment
                count = 0

                for lane in lanes:
                    count += lane_counts.get(lane, 0)

                # Calculate average speed if there are vehicles
                if count > 0 and lanes:
                    available_lanes = [l for l in lanes if l in lane_speeds]
                    all_vehicles = []
                    for lane in available_lanes:
                        all_vehicles.extend(self.world.eng.lane.getLastStepVehicleIDs(lane))
                    if all_vehicles:
                        avg_speed = sum(lane_speeds.get(v, 0.0) for v in all_vehicles) / len(all_vehicles)
                    else:
                        avg_speed = 0.0
                else:
                    avg_speed = 0.0

                # Normalize values
                state.append(count / 10.0)  # Normalize by max vehicles
                state.append(avg_speed / 50.0)  # Normalize by max speed (km/h)

        # Add current phase (one-hot)
        phase_onehot = [0] * len(self.intersection.phases)
        if current_phase < len(phase_onehot):
            phase_onehot[current_phase] = 1
        state.extend(phase_onehot)

        # Add phase duration ratio (normalized)
        max_phase_time = 60  # Assume max phase time is 60 seconds
        state.append(min(phase_duration / max_phase_time, 1.0))

        return np.array(state, dtype=np.float32)

    def _get_lane_data(self):
        """
        Get lane vehicle counts and speeds for the current simulation step.

        Returns:
            Tuple of (lane_counts dict, lane_speeds dict)
        """
        if isinstance(self.world, world_cityflow.World):
            lane_counts = self.world.eng.get_lane_vehicle_count()
            lane_speeds = self.world.eng.get_vehicle_speed()
        elif isinstance(self.world, world_sumo.World):
            import libsumo
            lane_counts = {}
            lane_speeds = {}
            for lane in self.world.all_lanes:
                lane_counts[lane] = 0
                # Get vehicles on this lane and their speeds
                vehicles = self.world.eng.lane.getLastStepVehicleIDs(lane)
                for vehicle_id in vehicles:
                    lane_counts[lane] += 1
                    if vehicle_id not in lane_speeds:
                        lane_speeds[vehicle_id] = self.world.eng.vehicle.getSpeed(vehicle_id)
            # Calculate average speed per lane if there are vehicles
            for lane, count in lane_counts.items():
                if count > 0:
                    avg_spd = sum(lane_speeds.get(v, 0.0) for v in self.world.eng.lane.getLastStepVehicleIDs(lane)) / count
                    lane_speeds[lane] = avg_spd
                else:
                    lane_speeds[lane] = 0.0
        return lane_counts, lane_speeds

    def _get_approach_lanes(self, approach):
        """
        Get incoming lanes for a specific approach.

        Args:
            approach: 'N', 'E', 'S', or 'W'

        Returns:
            List of lane IDs
        """
        roads = self.intersection.in_roads
        idx = self._approach_to_index(approach)

        if idx is None or idx >= len(roads):
            return []

        road = roads[idx]

        if isinstance(self.world, world_cityflow.World):
            lanes = []
            for i in range(len(road['lanes'])):
                lane_id = road['id'] + "_" + str(i)
                lanes.append(lane_id)
        elif isinstance(self.world, world_sumo.World):
            # For SUMO, use the pre-computed all_lanes and filter by road name
            # Get lane count for this road
            num_lanes = self.world.eng.edge.getLaneNumber(road)
            lanes = []
            for i in range(num_lanes):
                lane_id = road + "_" + str(i)
                lanes.append(lane_id)

        return lanes

    def _approach_to_index(self, approach):
        """Convert approach string to index."""
        approach_map = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        return approach_map.get(approach)

    def get_raw_state(self):
        """
        Get raw state without normalization for attacker policy.

        Returns:
            Dictionary with raw state components
        """
        state = {
            'vehicle_counts': [],
            'speeds': [],
            'current_phase': self.intersection.current_phase,
            'phase_duration': self.intersection.current_phase_time,
            'phases': self.intersection.phases
        }

        approaches = ['N', 'E', 'S', 'W']

        # Get lane data (works for both engines)
        lane_counts, lane_speeds = self._get_lane_data()

        for approach in approaches:
            lanes = self._get_approach_lanes(approach)
            seg_counts = []
            seg_speeds = []

            for seg_idx in range(self.num_segments):
                # Only process lanes that exist and are in our data
                available_lanes = [l for l in lanes if l in lane_counts]
                count = sum(lane_counts.get(l, 0) for l in available_lanes)

                # Calculate average speed from vehicles on these lanes
                if count > 0 and available_lanes:
                    all_vehicles = []
                    for lane in available_lanes:
                        all_vehicles.extend(self.world.eng.lane.getLastStepVehicleIDs(lane))
                    if all_vehicles:
                        avg_speed = sum(lane_speeds.get(v, 0.0) for v in all_vehicles) / len(all_vehicles)
                    else:
                        avg_speed = 0.0
                else:
                    avg_speed = 0.0

                seg_counts.append(count)
                seg_speeds.append(avg_speed)

            state['vehicle_counts'].append(seg_counts)
            state['speeds'].append(seg_speeds)

        return state
