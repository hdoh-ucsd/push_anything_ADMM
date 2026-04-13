
# Project Component Overview: Push Anything ADMM

This repository implements a **sample-based Contact-Implicit Model Predictive Control (CI-MPC)** framework, specifically **C3+**, using **PyDrake** for non-prehensile robotic manipulation. The system is designed to allow a 7-DOF Franka Emika Panda arm to move various objects via pushing by reasoning through contact physics in real-time.

### 📂 Root Directory

*   **`main.py`**: The primary entry point for the application. It orchestrates the simulation loop, initializes the PyDrake hardware station, instantiates the C3+ controller, and manages the real-time visualization through Meshcat.
*   **`setup.sh`**: A shell script for environment initialization and dependency installation.
*   **`environment.yml`**: The Conda environment configuration file defining necessary Python dependencies, including PyDrake and associated robotics libraries.

### 🧠 `control/` (Controller Logic)

This directory contains the mathematical core of the project, focusing on trajectory optimization through contact.
*   **MPC Implementation (C3/C3+)**: Implements the "Consensus Complementarity Control Plus" algorithm. It uses an MPPI-style sampling architecture to evaluate multiple potential future trajectories in parallel, allowing the robot to "hallucinate" and select the optimal sequence of pushes.
*   **Custom ADMM Solver**: A specialized Alternating Direction Method of Multipliers (ADMM) solver that computes rigid body dynamics through frictional contacts. It alternates between solving a smooth Quadratic Program (QP) for joint accelerations and a non-smooth proximal projection to enforce non-penetration and friction cone constraints.
*   **LCS Extraction**: Logic to dynamically extract the system's mass matrix, Coriolis terms, and **Linear Complementarity System (LCS)** contact Jacobians directly from the Drake simulation context.

### 🌍 `sim/` (Simulation Environment)

Contains the setup for the virtual workspace used to validate the controller.
*   **Physics Environment**: Defines the 3D simulation environment, which includes the Franka Panda arm, the planar workspace (table), and the dynamically manipulable rigid bodies.

### 📐 `models/` (Assets and Digital Twins)

*   **URDF & Mesh Files**: Stores the digital blueprints (Unified Robot Description Format) and 3D meshes for the Franka Panda arm and the various objects the robot is tasked to push.

### ⚙️ `config/` (Configuration)

*   **Parameter Management**: Contains configuration files defining critical system parameters such as MPC horizons, ADMM iteration limits, and robot impedance gains.

### 🛠️ `utils/` (Utilities)

*   **Math & Visualization**: Provides helper functions for mathematical operations and Meshcat rendering wrappers for the target goal and sampled trajectories.

***

**💡 Sidenote: What I am doing / What I am going to do**

*   **Current Progress:** Implementation in Python is slower than expected. 
*   **Future Roadmap:** Better Visualization on Motion Planning (Desired Goal), Understand what makes the implementation slow
***

### 📚 Citations
 H. Bui et al., **"Push Anything: Single- and Multi-Object Pushing From First Sight with Contact-Implicit MPC,"** *arXiv preprint arXiv:2510.19974v2*, 2025.

 A. Aydinoglu, A. Wei, W.-C. Huang, and M. Posa, **"Consensus complementarity control for multi-contact mpc,"** *IEEE Transactions on Robotics*, vol. 40, pp. 3879–3896, 2024.

 W. Yang and W. Jin, **"ContactSDF: Signed Distance Functions as Multi-Contact Models for Dexterous Manipulation,"** *arXiv preprint arXiv:2408.09612v2*, 2024.

 W. Heemels, J. M. Schumacher, and S. Weiland, **"Linear complementarity systems,"** *SIAM Journal on Applied Mathematics*, vol. 60, no. 4, pp. 1234–1269, 2000.
