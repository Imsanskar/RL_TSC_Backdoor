"""
Fake Vehicle Tracker - Perception-level attack tracking for CityFlow simulation.

This tracker records fake vehicle placements and integrates them into observation
getters to manipulate the victim agent's perception without actually adding vehicles
to the simulation.
"""

import numpy as np


class FakeVehicleTracker:
    """
    Tracks fake vehicles for perception-level attack.

    Does not modify simulation state directly but integrates counts into
    observation getters to poison victim agent's perception.
    """

    def __init__(self):
        # Track {lane_id: count} mapping
        self.fake_vehicles = {}
        # Track total fake vehicles per intersection
        self.intersections = {}  # {intersection_id: list of affected lanes}
        # Total injected counter for reward calculation
        self.total_injected = 0

    def insert_fake_vehicles(self, intersection_id, approach_name, vehicle_counts):
        """
        Record fake vehicles to be shown in victim agent's perception.

        Args:
            intersection_id: ID of the attacked intersection
            approach_name: Approach name ('N', 'E', 'S', or 'W')
            vehicle_counts: List of vehicle counts per segment (e.g., [2, 3, 1])

        Returns:
            Number of fake vehicles recorded
        """
        import cityflow
        intersection = None
        try:
            # Find intersection object to get approach lanes
            if hasattr(self._world, 'id2intersection'):
                intersection = self._world.id2intersection.get(intersection_id)

            if intersection is not None and approach_name in ['N', 'E', 'S', 'W']:
                approaches = ['N', 'E', 'S', 'W']
                approach_idx = {'N': 0, 'E': 1, 'S': 2, 'W': 3}.get(approach_name)

                if approach_idx is not None and approach_idx < len(intersection.in_roads):
                    road = intersection.in_roads[approach_idx]
                    lanes = []
                    for i in range(len(road['lanes'])):
                        lane_id = road['id'] + "_" + str(i)
                        lanes.append(lane_id)

                    total_fakes = 0
                    for seg_idx, count in enumerate(vehicle_counts):
                        if seg_idx < len(lanes) and count > 0:
                            lane = lanes[seg_idx]
                            # Record fake vehicles per lane
                            self.fake_vehicles[lane] = self.fake_vehicles.get(lane, 0) + count
                            total_fakes += count

                    # Track affected lanes for this intersection
                    if intersection_id not in self.intersections:
                        self.intersections[intersection_id] = []
                    for lane in lanes:
                        if lane not in self.intersections[intersection_id]:
                            self.intersections[intersection_id].append(lane)

                    self.total_injected += total_fakes
                    return total_fakes
            else:
                # If we can't get proper intersection info, just record counts per lane
                for seg_idx, count in enumerate(vehicle_counts):
                    if count > 0 and count < len(lanes):
                        lane = lanes[seg_idx]
                        self.fake_vehicles[lane] = self.fake_vehicles.get(lane, 0) + count
                return sum(c for c in vehicle_counts)
        except Exception:
            # Fallback: just accept all counts and assign to generic lanes
            self.total_injected += sum(vehicle_counts if isinstance(vehicle_counts, list) else [vehicle_counts])
            return sum(vehicle_counts) if isinstance(vehicle_counts, list) else vehicle_counts

    def reset_fake_vehicles(self, intersection_id=None):
        """
        Reset/remove all fake vehicles from tracking.

        Args:
            intersection_id: If provided, only reset for this intersection. None resets all.

        Returns:
            Number of fake vehicles cleared
        """
        removed = 0
        if intersection_id is not None and self.intersections.get(intersection_id):
            lanes_to_reset = self.intersections.pop(intersection_id, [])
            for lane in lanes_to_reset:
                if lane in self.fake_vehicles:
                    removed += self.fake_vehicles[lane]
                    del self.fake_vehicles[lane]
        else:
            # Reset all fake vehicle tracking
            for lane in list(self.fake_vehicles.keys()):
                removed += self.fake_vehicles[lane]
                del self.fake_vehicles[lane]
            self.intersections = {}

        return removed

    def get_fake_counts(self, lane_counts):
        """
        Get fake vehicle counts overlaid on real lane counts.

        Args:
            lane_counts: Current lane vehicle counts from simulation

        Returns:
            Dictionary mapping lane_id to total count (real + fake)
        """
        merged = {}
        for lane, count in lane_counts.items():
            fake_count = self.fake_vehicles.get(lane, 0)
            merged[lane] = count + fake_count
        return merged

    def get_fake_vehicle_count(self):
        """Get total number of fake vehicles tracked."""
        return self.total_injected

    @property
    def has_fake_vehicles(self):
        """Check if any fake vehicles are being tracked."""
        return len(self.fake_vehicles) > 0 or self.total_injected > 0