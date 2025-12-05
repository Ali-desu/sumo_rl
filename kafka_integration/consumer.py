import json
import time
import pickle
import numpy as np
from kafka import KafkaConsumer, KafkaProducer

from app.config.config import INTERSECTIONS


def main():
    # Load Q-tables for each agent
    q_tables = {}
    for name in INTERSECTIONS.keys():
        with open(f'models/{name}_q_table.pkl', 'rb') as f:
            q_tables[name] = pickle.load(f)

    # Kafka consumer for junction_state
    consumer = KafkaConsumer(
        'junction_state',
        bootstrap_servers=['localhost:9092'],
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='latest'  # Start from latest messages
    )

    # Kafka producer for action
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    latest_states = {}

    print("Consumer started, listening to junction_state...")

    while True:
        # Poll for new messages (states)
        messages = consumer.poll(timeout_ms=1000)  # Poll for 1 second
        for topic_partition, msgs in messages.items():
            for msg in msgs:
                data = msg.value
                node = data['node']
                state = tuple(data['state'])  # Convert list back to tuple
                latest_states[node] = state
                print(f"Received state for {node}: {state}")

        # Every 10 seconds, process latest states and publish actions
        time.sleep(10)
        if latest_states:
            actions = {}
            for node, state in latest_states.items():
                q_table = q_tables.get(node, {})
                if state in q_table:
                    action = int(np.argmax(q_table[state]))  # Greedy action
                else:
                    action = 0  # Default action if state not seen
                actions[node] = action

            message = {
                "actions": actions,
                "timestamp": time.time()
            }
            producer.send('action', message)
            print(f"Published actions: {actions}")
            # Clear latest_states after publishing, or keep for next cycle
            latest_states.clear()


if __name__ == "__main__":
    main()
