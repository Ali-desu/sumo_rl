import pickle
import numpy as np

def load_model(file_path):
    """Load a machine learning model from a pickle file."""
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Load the model
loaded_model = load_model('models/Node2_q_table.pkl')

state = (5, 3, 0, 2,5, 3, 0, 2,5, 3, 0, 2, 120, 80, 0, 45,120, 80, 0, 45,120, 80, 0, 45, 1)

action = np.argmax(loaded_model[state])

print(f"Chosen action for state {state}: {action}")#REF20251205512