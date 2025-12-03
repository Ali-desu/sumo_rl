"""
multi_intersection_qlearning_with_waiting.py

- One agent per intersection (Node2 and Node3).
- Uses laneAreaDetector IDs (the 'id' attribute in your detector XML).
- State includes per-direction queue length and waiting time, plus current phase.
- Reward = - (total_waiting_time + total_queue_length)
- Action space (per-agent): 0 = keep current phase, 1 = switch to next phase (only if min green satisfied).
"""

import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# SUMO / TraCI bootstrap check
# -----------------------------
if 'SUMO_HOME' not in os.environ:
    sys.exit("ERROR: Please set the SUMO_HOME environment variable to your SUMO installation folder.")

SUMO_TOOLS = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(SUMO_TOOLS)
import traci

# -----------------------------
# Config parameters
# -----------------------------
sumo_binary = "sumo-gui"
sumo_cfg = "RL.sumocfg"          # your SUMO configuration
STEP_LENGTH = 1.0
TOTAL_STEPS = 1000
MIN_GREEN_SEC = 10
MIN_GREEN_STEPS = int(MIN_GREEN_SEC / STEP_LENGTH)

ALPHA = 0.2
GAMMA = 0.9
EPSILON_START = 0.3
EPSILON_END = 0.05
EPSILON_DECAY = (EPSILON_START - EPSILON_END) / (TOTAL_STEPS * 0.9)
ACTIONS = [0, 1]  # 0: keep, 1: switch

# -----------------------------
# Intersections config (detector 'id's)
# -----------------------------
INTERSECTIONS = {
    "I_Node2": {
        "tls_id": "Node2",
        "detector_groups": [
            ["Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2"],
            ["Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2"],
            ["e2_0", "e2_1", "e2_2"],
            ["e2_3", "e2_4", "e2_5"]
        ]
    },
    "I_Node3": {
        "tls_id": "Node3",
        "detector_groups": [
            ["e2_10", "e2_11", "e2_9"],
            ["e2_18", "e2_19", "e2_20"],
            ["e2_6", "e2_7", "e2_8"],
            ["e2_15", "e2_16", "e2_17"]
        ]
    }
}

# -----------------------------
# Helper functions
# -----------------------------
def safe_get_vehicle_number(det_id):
    try:
        return traci.lanearea.getLastStepVehicleNumber(det_id)
    except:
        return 0

def safe_get_vehicle_ids(det_id):
    try:
        vids = traci.lanearea.getLastStepVehicleIDs(det_id)
        return tuple(vids) if vids else tuple()
    except:
        return tuple()

def get_group_queue_and_wait(detector_group):
    q_sum = 0
    wait_sum = 0.0
    seen = set()
    for det in detector_group:
        q = safe_get_vehicle_number(det)
        q_sum += q
        vids = safe_get_vehicle_ids(det)
        for vid in vids:
            if vid in seen:
                continue
            seen.add(vid)
            try:
                wait_sum += traci.vehicle.getWaitingTime(vid)
            except:
                wait_sum += 0.0
    return q_sum, wait_sum

def get_state_for_intersection(cfg):
    q_list = []
    w_list = []
    for group in cfg["detector_groups"]:
        q, w = get_group_queue_and_wait(group)
        q_list.append(int(q))
        w_list.append(int(round(w)))
    try:
        phase = traci.trafficlight.getPhase(cfg["tls_id"])
    except:
        phase = -1
    return tuple(q_list + w_list + [phase])

# -----------------------------
# Agent class
# -----------------------------
class IntersectionAgent:
    def __init__(self, name, cfg):
        self.name = name
        self.tls_id = cfg["tls_id"]
        self.detector_groups = cfg["detector_groups"]
        self.q_table = {}
        self.last_switch_step = -MIN_GREEN_STEPS
        self.epsilon = EPSILON_START
        self.cumulative_reward = 0.0
        self.queue_history = []
        self.avg_wait_history = []
        self.dir_queue_history = []

    def ensure_state(self, s):
        if s not in self.q_table:
            self.q_table[s] = np.zeros(len(ACTIONS))

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        self.ensure_state(state)
        return int(np.argmax(self.q_table[state]))

    def can_switch(self, current_step):
        return (current_step - self.last_switch_step) >= MIN_GREEN_STEPS

    def apply_action(self, action, current_step):
        if action == 0 or not self.can_switch(current_step):
            return
        try:
            logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
            num_phases = len(logic.phases)
            cur_phase = traci.trafficlight.getPhase(self.tls_id)
            next_phase = (cur_phase + 1) % num_phases
            traci.trafficlight.setPhase(self.tls_id, next_phase)
            self.last_switch_step = current_step
        except Exception as e:
            print(f"[{self.name}] apply_action exception: {e}")

    def compute_reward_and_stats(self):
        total_q = 0
        total_w = 0.0
        for group in self.detector_groups:
            q, w = get_group_queue_and_wait(group)
            total_q += q
            total_w += w
        reward = - (total_q + total_w)
        return reward, total_q, total_w

    def update_q_table(self, old_state, action, reward, new_state):
        self.ensure_state(old_state)
        self.ensure_state(new_state)
        old_q = self.q_table[old_state][action]
        best_next = np.max(self.q_table[new_state])
        self.q_table[old_state][action] = old_q + ALPHA * (reward + GAMMA * best_next - old_q)

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon - EPSILON_DECAY)

# -----------------------------
# Main simulation
# -----------------------------
def main():
    agents = {name: IntersectionAgent(name, cfg) for name, cfg in INTERSECTIONS.items()}

    sumo_cmd = [sumo_binary, "-c", sumo_cfg, "--step-length", str(STEP_LENGTH)]
    print("Starting SUMO with:", " ".join(sumo_cmd))
    traci.start(sumo_cmd)

    last_decisions = {name: None for name in agents.keys()}

    for step in range(TOTAL_STEPS):
        # observe + act
        for name, agent in agents.items():
            state = get_state_for_intersection({
                "detector_groups": agent.detector_groups,
                "tls_id": agent.tls_id
            })
            action = agent.select_action(state)
            agent.apply_action(action, step)
            last_decisions[name] = (state, action)

        traci.simulationStep()

        # update Q-table and histories
        for name, agent in agents.items():
            if last_decisions[name] is None:
                continue
            old_state, action = last_decisions[name]
            new_state = get_state_for_intersection({
                "detector_groups": agent.detector_groups,
                "tls_id": agent.tls_id
            })
            r, total_q, total_w = agent.compute_reward_and_stats()
            agent.cumulative_reward += r
            agent.update_q_table(old_state, action, r, new_state)
            agent.reward_history = getattr(agent, "reward_history", [])
            agent.reward_history.append(agent.cumulative_reward)
            agent.queue_history.append(total_q)
            agent.avg_wait_history.append(total_w / max(total_q, 1))
            # per-direction queues
            dir_queues = [get_group_queue_and_wait(g)[0] for g in agent.detector_groups]
            agent.dir_queue_history.append(dir_queues)
            agent.decay_epsilon()

        if step % 100 == 0:
            print(f"Step {step}")
            for name, agent in agents.items():
                print(f"[{name}] eps={agent.epsilon:.3f} cum_reward={agent.cumulative_reward:.1f}")

    traci.close()

    # -----------------------------
    # Plot results
    # -----------------------------
    for name, agent in agents.items():
        steps = len(agent.queue_history)
        if steps == 0:
            continue
        t = np.arange(steps)
        plt.figure(figsize=(14,6))

        # Total + per direction queues
        plt.subplot(2,1,1)
        plt.plot(t, agent.queue_history, label="Total Queue", color="black")
        dir_arr = np.array(agent.dir_queue_history)
        for i in range(dir_arr.shape[1]):
            plt.plot(t, dir_arr[:,i], linestyle='--', alpha=0.7, label=f"Dir {i}")
        plt.title(f"{name} lane occupancy over time")
        plt.xlabel("Step")
        plt.ylabel("Vehicles")
        plt.legend()

        # Average waiting time
        plt.subplot(2,1,2)
        plt.plot(t, agent.avg_wait_history, color="orange")
        plt.title(f"{name} average waiting time per vehicle")
        plt.xlabel("Step")
        plt.ylabel("Seconds")

        plt.tight_layout()
        plt.show()

    # Q-table sizes
    for name, agent in agents.items():
        print(f"{name} final Q-table size: {len(agent.q_table)}")

if __name__ == "__main__":
    main()
