# Multi-Agent Traffic Control with Q-Learning and SUMO

This project implements a **multi-agent reinforcement learning (RL) system** for optimizing traffic light control at urban intersections using Q-learning algorithms integrated with SUMO (Simulation of Urban Mobility). The system simulates traffic scenarios where multiple RL agents coordinate to minimize vehicle queues and waiting times, improving overall traffic flow efficiency.

## Features

- **Multi-Agent Q-Learning**: Independent RL agents control traffic lights at different intersections, learning optimal phase-switching strategies.
- **SUMO Integration**: Real-time traffic simulation using SUMO's TraCI interface for accurate vehicle dynamics and detector data.
- **Adaptive Learning**: Epsilon-greedy exploration with decay enables agents to explore actions initially and exploit learned policies over time.
- **Performance Tracking**: Comprehensive logging of queue lengths, waiting times, rewards, and cumulative performance metrics.
- **Visualization**: Automatic plotting of simulation results using Matplotlib for analysis and comparison.
- **Configurable Scenarios**: Easily define new intersections, detector groups, and hyperparameters via configuration files.
- **Modular Design**: Clean separation of concerns with dedicated modules for agents, utilities, configuration, and simulation logic.

## Installation

### Prerequisites
- **Python 3.8+**: Ensure Python is installed on your system.
- **SUMO (Simulation of Urban Mobility)**: Download and install SUMO from the [official website](https://www.eclipse.org/sumo/). Set the `SUMO_HOME` environment variable to the SUMO installation directory (e.g., `C:\Program Files\SUMO` on Windows).

### Requirements
Install the required Python packages using the provided `requirements.txt` file:
```
pip install -r requirements.txt
```

### Setup

1. Clone or download this repository to your local machine.
2. Navigate to the project directory: `cd multi_agents`
3. Ensure SUMO is properly installed and `SUMO_HOME` is set.
4. Verify the SUMO configuration files in `sumo_files/` are correctly referenced (e.g., `RL.sumocfg` points to valid network and route files).

## Usage

### Running the Simulation
1. Open a terminal in the project root directory.
2. Execute the main script:
   ```
   python app/main.py
   ```
3. The simulation will start SUMO (with GUI by default) and run for 1000 steps.
4. Monitor progress in the console: It displays current step, epsilon values, and cumulative rewards for each agent.
5. Upon completion, Matplotlib plots will appear showing queue lengths and average waiting times per intersection.

### Customizing the Simulation
- **Headless Mode**: Change `SUMO_BINARY = "sumo"` in `app/config/config.py` to run without GUI.
- **Adjust Parameters**: Modify hyperparameters in `app/config/config.py` (e.g., learning rate, total steps, epsilon decay).
- **Add Intersections**: Define new intersections in the `INTERSECTIONS` dictionary with their TLS IDs and detector groups.
- **Change Scenarios**: Update SUMO files in `sumo_files/` (e.g., network layout in `RL.net.xml`, routes in `RL.rou.xml`).

### Expected Output
- Console logs every 100 steps with agent performance.
- Final Q-table sizes for each agent.
- Plots for each intersection: Total queue length over time, directional queues, and average waiting time.

## Project Structure

### `app/` Directory (Main Application Logic)
- **`main.py`**: Entry point of the application. Initializes agents, starts the SUMO simulation, runs the RL loop (act-observe-learn), handles interruptions, and generates result plots.
- **`agents/agent.py`**: Defines the `QLearningAgent` class. Each agent manages its Q-table, chooses actions (keep or switch traffic light phase), computes rewards based on queue lengths and waiting times, and updates its policy.
- **`config/config.py`**: Contains all configuration constants, including SUMO settings, RL hyperparameters (alpha, gamma, epsilon), and intersection definitions with detector groups.
- **`utils/` Subdirectory**:
  - **`utils.py`**: Utility functions for extracting simulation states (vehicle counts per detector group) and calculating queue lengths and waiting times.
  - **`logger.py`**: Handles logging and history tracking for performance metrics (queues, waiting times, rewards) over simulation steps.

### `sumo_files/` Directory (SUMO Simulation Assets)
- **`RL.net.xml`**: Defines the road network, including junctions, lanes, and connections for the traffic simulation.
- **`RL.rou.xml`**: Specifies vehicle routes, traffic flows, and departure times to simulate realistic traffic patterns.
- **`RL.add.xml`**: Additional simulation elements, such as induction loops (detectors) for monitoring vehicle presence and speed.
- **`RL.sumocfg`**: SUMO configuration file that ties together the network, routes, and additional files to run the simulation.
- **`e2_*.xml` (0-20)`**: Individual detector configuration files for collecting data at specific edges or lanes (used for state observation and reward calculation).
- **`Q_Rl.py`**: Standalone script demonstrating Q-learning integration with SUMO (possibly for testing or comparison).
- **`Normal.py`**: Baseline simulation script without RL, for comparing against learned policies.
- **`q_table.pkl`**: Serialized Q-table file storing learned state-action values from previous training runs (can be loaded for continued learning or evaluation).
- **`RL.netecfg`**: Additional network configuration file (less commonly used).

## Configuration

Key settings are centralized in `app/config/config.py`:
- **Simulation Parameters**: Total steps (1000), step length (1.0s), SUMO binary and config file paths.
- **Traffic Light Constraints**: Minimum green time (10 seconds) to prevent rapid switching.
- **RL Hyperparameters**:
  - `ALPHA` (0.2): Learning rate for Q-value updates.
  - `GAMMA` (0.9): Discount factor for future rewards.
  - `EPSILON_START` (0.3), `EPSILON_END` (0.05), `EPSILON_DECAY`: Exploration parameters.
- **Actions**: [0, 1] representing keep current phase or switch to next.
- **Intersections**: Dictionary defining controlled intersections (e.g., "Node2", "Node3") with TLS IDs and associated detector groups for state observation.

Modify these values to experiment with different learning behaviors or scenarios.

## How It Works

### Simulation Flow
1. **Initialization**: Agents are created for each defined intersection, Q-tables are initialized, and SUMO simulation is started.
2. **Act Phase**: For each agent, the current state (discretized vehicle counts per direction) is observed, an action is chosen (epsilon-greedy), and applied if valid (respecting minimum green time).
3. **Simulation Advance**: SUMO steps forward by one time unit, vehicles move, and detectors collect new data.
4. **Observe & Learn**: Agents compute rewards (negative of total queue + waiting time), update Q-values using the Bellman equation, record metrics, and decay epsilon.
5. **Repeat**: Cycles continue until the total steps are reached.
6. **Termination**: Simulation closes, results are plotted, and Q-table sizes are reported.

### RL Mechanics
- **States**: Represented as tuples of vehicle counts across detector groups (e.g., queues per direction).
- **Actions**: Binary choice to maintain or change traffic light phase.
- **Rewards**: Penalize congestion (queue length + accumulated waiting time).
- **Learning**: Q-learning updates maximize expected future rewards, enabling agents to learn coordinated timing strategies.

## Results and Visualization

- **Console Output**: Real-time updates on agent progress, including epsilon values and cumulative rewards.
- **Plots**: Generated for each intersection using Matplotlib:
  - **Queue Lengths**: Total queue and per-direction queues over time.
  - **Average Waiting Time**: Per-vehicle waiting time progression.
- **Q-Table Analysis**: Final sizes indicate the number of explored states, reflecting learning complexity.

These outputs help evaluate agent performance, compare intersections, and identify areas for hyperparameter tuning.

## Dependencies

- **Python Libraries**:
  - `numpy`: Numerical computations and Q-table management.
  - `matplotlib`: Plotting simulation results.
  - `traci`: SUMO's Python interface for simulation control.
- **External Tools**:
  - **SUMO**: Core simulation engine (version 1.10+ recommended).
- Ensure all dependencies are installed and SUMO is accessible via `SUMO_HOME`.

## Contributing and Notes

- **Modularity**: The project is designed for easy extension—add new agent types, reward functions, or state representations in the respective modules.
- **Experimentation**: Use the configuration file to test different hyperparameters or scenarios without code changes.
- **Performance**: For large simulations, consider headless SUMO mode and monitor memory usage due to Q-table growth.
- **Troubleshooting**: If SUMO fails to start, verify `SUMO_HOME` and file paths. Ensure detector IDs in config match those in SUMO files.
- **Future Enhancements**: Potential additions include multi-objective rewards (e.g., emissions), advanced RL algorithms (e.g., Deep Q-Networks), or real-world data integration.

For questions or contributions, refer to the code comments or experiment with the provided scripts. This system demonstrates the power of RL in optimizing complex, dynamic systems like urban traffic.
