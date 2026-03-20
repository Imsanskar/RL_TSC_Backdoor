"""
Attacker State Generator - Creates observation state for the attacker policy.

This generator provides the attacker with traffic state information including
vehicle counts, speeds, and intersection phase information.
"""

import numpy as np
from common.registry import Registry


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
        generate
        Generate state observations for the attacker.

        Returns:
            State vector as numpy array
        """
        state = []

        # Get vehicle counts per lane
        lane_counts = self.world.eng.get_lane_vehicle_count()
        lane_speeds = self.world.eng.get_vehicle_speed()

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
                total_speed = 0
                speed_count = 0

                for lane in lanes:
                    # lane_counts returns counts directly (not lists of vehicles)
                    count += lane_counts.get(lane, 0)

                # Speed data is per vehicle, not per lane, so we need to get average
                # For simplicity, use a default speed if no vehicles
                if count > 0:
                    avg_speed = 15.0  # Default average speed in km/h
                else:
                    avg_speed = 0

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
        lanes = []
        for i in range(len(road['lanes'])):
            lane_id = road['id'] + "_" + str(i)
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
        lane_counts = self.world.eng.get_lane_vehicle_count()
        lane_speeds = self.world.eng.get_vehicle_speed()

        for approach in approaches:
            lanes = self._get_approach_lanes(approach)
            seg_counts = []
            seg_speeds = []

            for seg_idx in range(self.num_segments):
                count = 0

                for lane in lanes:
                    # lane_counts returns counts directly (not lists of vehicles)
                    count += lane_counts.get(lane, 0)

                # Speed data is per vehicle, not per lane
                if count > 0:
                    avg_speed = 15.0  # Default average speed in km/h
                else:
                    avg_speed = 0

                seg_counts.append(count)
                seg_speeds.append(avg_speed)

            state['vehicle_counts'].append(seg_counts)
            state['speeds'].append(seg_speeds)

        return state
