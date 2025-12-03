"""
Simple history logger for plotting and analysis.
"""

from typing import List

class HistoryLogger:
    def __init__(self):
        self.queue_total: List[int] = []
        self.avg_wait: List[float] = []
        self.dir_queues: List[List[int]] = []   # list of lists (one per direction)
        self.rewards: List[float] = []
        self.cum_reward: List[float] = []

    def record(self, total_q: int, total_w: float, dir_q: List[int], reward: float, cum_reward: float):
        self.queue_total.append(total_q)
        self.avg_wait.append(total_w / max(total_q, 1))
        self.dir_queues.append(dir_q[:])
        self.rewards.append(reward)
        self.cum_reward.append(cum_reward)