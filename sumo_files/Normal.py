"""
static_traffic_per_node.py

- Runs SUMO with static traffic lights (no RL).
- Collects per-intersection queue and average waiting time.
- Produces plots similar to RL script for Node2 and Node3.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# SUMO / TraCI bootstrap
# -----------------------------
if 'SUMO_HOME' not in os.environ:
    sys.exit("ERROR: Please set the SUMO_HOME environment variable to your SUMO installation folder.")

SUMO_TOOLS = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(SUMO_TOOLS)
import traci

# -----------------------------
# Config
# -----------------------------
sumo_binary = "sumo-gui"
sumo_cfg = "RL.sumocfg"
STEP_LENGTH = 1.0
TOTAL_STEPS = 1000

# -----------------------------
# Intersections and detectors
# -----------------------------
INTERSECTIONS = {
    "Node2": [
        "Node1_2_EB_0","Node1_2_EB_1","Node1_2_EB_2",
        "Node2_7_SB_0","Node2_7_SB_1","Node2_7_SB_2",
        "e2_0","e2_1","e2_2",
        "e2_3","e2_4","e2_5"
    ],
    "Node3": [
        "e2_10","e2_11","e2_9",
        "e2_18","e2_19","e2_20",
        "e2_6","e2_7","e2_8",
        "e2_15","e2_16","e2_17"
    ]
}

# -----------------------------
# Helper functions
# -----------------------------
def safe_get_vehicle_number(det_id):
    try:
        return traci.lanearea.getLastStepVehicleNumber(det_id)
    except Exception:
        return 0

def safe_get_vehicle_ids(det_id):
    try:
        vids = traci.lanearea.getLastStepVehicleIDs(det_id)
        return vids if vids is not None else tuple()
    except Exception:
        return tuple()

def compute_node_stats(detector_ids):
    total_queue = 0
    total_wait = 0.0
    seen = set()
    for det in detector_ids:
        q = safe_get_vehicle_number(det)
        total_queue += q
        for vid in safe_get_vehicle_ids(det):
            if vid in seen:
                continue
            seen.add(vid)
            try:
                total_wait += traci.vehicle.getWaitingTime(vid)
            except Exception:
                total_wait += 0.0
    avg_wait = total_wait / max(total_queue, 1)
    return total_queue, avg_wait

# -----------------------------
# Main simulation
# -----------------------------
def main():
    # Start SUMO
    sumo_cmd = [sumo_binary, "-c", sumo_cfg, "--step-length", str(STEP_LENGTH)]
    print("Starting SUMO with:", " ".join(sumo_cmd))
    traci.start(sumo_cmd)

    # Histories per node
    histories = {node: {"queue": [], "avg_wait": []} for node in INTERSECTIONS.keys()}

    start_time = time.time()
    for step in range(TOTAL_STEPS):
        traci.simulationStep()

        for node, dets in INTERSECTIONS.items():
            q, avg_w = compute_node_stats(dets)
            histories[node]["queue"].append(q)
            histories[node]["avg_wait"].append(avg_w)

        if step % 100 == 0:
            print(f"Step {step}: Node2 queue={histories['Node2']['queue'][-1]}, avg_wait={histories['Node2']['avg_wait'][-1]:.2f} | "
                  f"Node3 queue={histories['Node3']['queue'][-1]}, avg_wait={histories['Node3']['avg_wait'][-1]:.2f}")

    traci.close()
    duration = time.time() - start_time
    print(f"Simulation finished in {duration:.1f}s")

    # -----------------------------
    # Plot per-node stats
    # -----------------------------
    for node, data in histories.items():
        t = np.arange(TOTAL_STEPS)
        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        plt.plot(t, data["queue"])
        plt.title(f"{node} total queue (vehicles)")
        plt.xlabel("Step")
        plt.ylabel("Vehicles")
        plt.grid(True)

        plt.subplot(1,2,2)
        plt.plot(t, data["avg_wait"], color='orange')
        plt.title(f"{node} average waiting time (s)")
        plt.xlabel("Step")
        plt.ylabel("Seconds")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
