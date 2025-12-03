"""
Q-Learning agent for a single intersection.
"""

import random
from typing import List, Tuple, Dict

import numpy as np
import traci
from traci.exceptions import TraCIException

from app.config.config import Config
from app.utils.utils import get_group_queue_and_wait
from app.utils.logger import HistoryLogger


class QLearningAgent:
    def __init__(self, name: str, tls_id: str, detector_groups: List[List[str]]):
        self.name = name
        self.tls_id = tls_id
        self.detector_groups = detector_groups

        self.q_table: Dict[Tuple, np.ndarray] = {}
        self.last_state: Tuple | None = None
        self.last_action: int | None = None
        self.last_switch_step = -Config.MIN_GREEN_STEPS

        self.epsilon = Config.EPSILON_START
        self.cumulative_reward = 0.0

        self.history = HistoryLogger()

    def _ensure_q_values(self, state: Tuple):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(Config.ACTIONS))

    def choose_action(self, state: Tuple) -> int:
        self._ensure_q_values(state)
        if random.random() < self.epsilon:
            return random.choice(Config.ACTIONS)
        return int(np.argmax(self.q_table[state]))

    def can_switch(self, current_step: int) -> bool:
        return (current_step - self.last_switch_step) >= Config.MIN_GREEN_STEPS

    def apply_action(self, action: int, current_step: int):
        if action == 0 or not self.can_switch(current_step):
            return

        try:
            logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
            cur_phase = traci.trafficlight.getPhase(self.tls_id)
            next_phase = (cur_phase + 1) % len(logic.phases)
            traci.trafficlight.setPhase(self.tls_id, next_phase)
            self.last_switch_step = current_step
        except TraCIException as e:
            print(f"[{self.name}] Error switching phase: {e}")

    def compute_reward_and_stats(self) -> Tuple[float, int, float, List[int]]:
        total_q = 0
        total_w = 0.0
        dir_queues = []

         # list to hold queue per direction

        for group in self.detector_groups:
            q, w = get_group_queue_and_wait(group)
            total_q += q
            total_w += w
            dir_queues.append(q)

        reward = -(total_q + total_w)
        return reward, total_q, total_w, dir_queues

    def update(self, new_state: Tuple, reward: float):
        if self.last_state is None or self.last_action is None:
            self.last_state = new_state
            return

        self._ensure_q_values(self.last_state)
        self._ensure_q_values(new_state)

        old_q = self.q_table[self.last_state][self.last_action]
        best_next = np.max(self.q_table[new_state])
        new_q = old_q + Config.ALPHA * (reward + Config.GAMMA * best_next - old_q)
        self.q_table[self.last_state][self.last_action] = new_q

        self.last_state = new_state

    def record_step(self, reward: float, total_q: int, total_w: float, dir_queues: List[int]):
        self.cumulative_reward += reward
        self.history.record(total_q, total_w, dir_queues, reward, self.cumulative_reward)

    def decay_epsilon(self):
        self.epsilon = max(Config.EPSILON_END, self.epsilon - Config.EPSILON_DECAY)