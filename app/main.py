"""
Main simulation script.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import traci
import pickle

from app.config.config import Config, INTERSECTIONS
from  app.utils.utils import get_state
from app.agents.agent import QLearningAgent


def create_agents() -> dict[str, QLearningAgent]:
    return {
        name: QLearningAgent(name=name, tls_id=cfg["tls_id"], detector_groups=cfg["detector_groups"])
        for name, cfg in INTERSECTIONS.items()
    }


def plot_results(agents: dict[str, QLearningAgent]):
    for name, agent in agents.items():
        h = agent.history
        if not h.queue_total:
            continue

        t = np.arange(len(h.queue_total))
        dir_arr = np.array(h.dir_queues)

        plt.figure(figsize=(14, 8))

        # Queue lengths
        plt.subplot(2, 1, 1)
        plt.plot(t, h.queue_total, label="Total Queue", color="black", linewidth=2)
        for i in range(dir_arr.shape[1]):
            plt.plot(t, dir_arr[:, i], linestyle="--", alpha=0.7, label=f"Direction {i}")
        plt.title(f"{name} - Queue Lengths")
        plt.ylabel("Vehicles")
        plt.legend()
        plt.grid(True)

        # Average waiting time
        plt.subplot(2, 1, 2)
        plt.plot(t, h.avg_wait, color="orange")
        plt.title(f"{name} - Avg Waiting Time per Vehicle")
        plt.xlabel("Simulation Step")
        plt.ylabel("Seconds")
        plt.grid(True)

        plt.tight_layout()
        plt.show()


def main():
    if 'SUMO_HOME' not in os.environ:
        sys.exit("Please set the SUMO_HOME environment variable.")

    agents = create_agents()

    sumo_cmd = [
        Config.SUMO_BINARY,
        "-c", Config.SUMO_CFG,
        "--step-length", str(Config.STEP_LENGTH),
    ]
    print("Starting SUMO:", " ".join(sumo_cmd))
    traci.start(sumo_cmd)

    step = 0
    try:
        while step < Config.TOTAL_STEPS:
            # ----- Act -----
            for agent in agents.values():
                state = get_state(agent.tls_id, agent.detector_groups)
                action = agent.choose_action(state)
                agent.apply_action(action, step)

                # Prepare for learning on next observation
                if agent.last_state is not None:
                    agent.last_action = action
                else:
                    agent.last_state = state
                    agent.last_action = action

            # ----- Advance simulation -----
            traci.simulationStep()
            step += 1

            #

            # ----- Observe & Learn -----
            for agent in agents.values():
                new_state = get_state(agent.tls_id, agent.detector_groups)
                reward, total_q, total_w, dir_q = agent.compute_reward_and_stats()

                agent.update(new_state, reward)
                agent.record_step(reward, total_q, total_w, dir_q)
                agent.decay_epsilon()

            if step % 100 == 0:
                print(f"\nStep {step}/{Config.TOTAL_STEPS}")
                for name, ag in agents.items():
                    print(f"  [{name}] ε={ag.epsilon:.3f} | cum_reward={ag.cumulative_reward:.1f}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        traci.close()

    print("\nSimulation finished.")
    for name, ag in agents.items():
        print(f"{name} final Q-table size: {len(ag.q_table)} states")

    # Save Q-tables for each agent
    for name, agent in agents.items():
        with open(f'models/{name}_q_table.pkl', 'wb') as f:
            pickle.dump(agent.q_table, f)
    print("Q-tables saved to models/ directory.")

    plot_results(agents)


if __name__ == "__main__":
    main()
