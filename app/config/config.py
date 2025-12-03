"""
Configuration constants and intersection definitions.
Easy to tweak hyperparameters or add new intersections.
"""

from typing import Dict, List, Any

class Config:
    # Simulation
    SUMO_BINARY = "sumo-gui"          # use "sumo" for headless
    SUMO_CFG = "sumo_files/RL.sumocfg"
    STEP_LENGTH = 1.0
    TOTAL_STEPS = 1000

    # Traffic-light constraints
    MIN_GREEN_SEC = 10
    MIN_GREEN_STEPS = int(MIN_GREEN_SEC / STEP_LENGTH)

    # RL hyperparameters
    ALPHA = 0.2          # learning rate
    GAMMA = 0.9          # discount factor
    EPSILON_START = 0.3
    EPSILON_END = 0.05
    EPSILON_DECAY = (EPSILON_START - EPSILON_END) / (TOTAL_STEPS * 0.9)

    ACTIONS = [0, 1]     # 0: keep phase, 1: switch to next phase


# Define all controlled intersections here
INTERSECTIONS: Dict[str, Dict[str, Any]] = {
    "Node2": {
        "tls_id": "Node2",
        "detector_groups": [
            ["Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2"],
            ["Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2"],
            ["e2_0", "e2_1", "e2_2"],
            ["e2_3", "e2_4", "e2_5"],
        ],
    },
    "Node3": {
        "tls_id": "Node3",
        "detector_groups": [
            ["e2_10", "e2_11", "e2_9"],
            ["e2_18", "e2_19", "e2_20"],
            ["e2_6", "e2_7", "e2_8"],
            ["e2_15", "e2_16", "e2_17"],
        ],
    },
}