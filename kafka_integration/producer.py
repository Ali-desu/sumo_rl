import os
import sys
import json
import time
from kafka import KafkaProducer

import traci

from app.config.config import Config, INTERSECTIONS
from app.utils.utils import get_state
from app.agents.agent import QLearningAgent


def create_agents() -> dict[str, QLearningAgent]:
    return {
        name: QLearningAgent(name=name, tls_id=cfg["tls_id"], detector_groups=cfg["detector_groups"])
        for name, cfg in INTERSECTIONS.items()
    }


def main():
    if 'SUMO_HOME' not in os.environ:
        sys.exit("Please set the SUMO_HOME environment variable.")

    # Kafka producer
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

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
            # Get state for each agent and publish
            for name, agent in agents.items():
                state = get_state(agent.tls_id, agent.detector_groups)
                message = {
                    "node": name,
                    "state": list(state),  # Convert tuple to list for JSON
                    "step": step
                }
                producer.send('junction_state', message)
                print(f"Published state for {name} at step {step}")

            # Advance simulation
            traci.simulationStep()
            step += 1

            # Optional: sleep to control publishing rate, e.g., every 1 second
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        traci.close()
        producer.close()


if __name__ == "__main__":
    main()
