"""
Utility functions for interacting with SUMO/TraCI.
"""

from typing import List, Tuple
import traci
from traci.exceptions import TraCIException


def safe_lanearea_value(func, det_id, default=0):
    try:
        return func(det_id)
    except TraCIException:
        return default


def get_group_queue_and_wait(detector_group: List[str]) -> Tuple[int, float]:
    """Return (total vehicles, total waiting time) for a group of detectors."""
    queue_sum = 0
    wait_sum = 0.0
    seen_vehicles = set()

    for det_id in detector_group:
        queue_sum += safe_lanearea_value(traci.lanearea.getLastStepVehicleNumber, det_id, 0)
        vehicles = safe_lanearea_value(traci.lanearea.getLastStepVehicleIDs, det_id, ())

        for vid in vehicles:
            if vid in seen_vehicles:
                continue
            seen_vehicles.add(vid)
            wait_sum += safe_lanearea_value(traci.vehicle.getWaitingTime, vid, 0.0)

    return queue_sum, wait_sum


def get_state(tls_id: str, detector_groups: List[List[str]]) -> Tuple:
    """State = (q0, q1, ..., w0, w1, ..., current_phase)"""
    queues = []
    waits = []
    for group in detector_groups:
        q, w = get_group_queue_and_wait(group)
        queues.append(int(q))
        waits.append(int(round(w)))

    try:
        phase = traci.trafficlight.getPhase(tls_id)
    except TraCIException:
        phase = -1

    return tuple(queues + waits + [phase])