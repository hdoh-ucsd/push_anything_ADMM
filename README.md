```markdown
# Push Anything: Teaching Robots to Nudge

## 🤖 What is this project?
This project is a computer program that teaches a robotic arm (specifically a **Franka Emika Panda**) how to move objects on a table by **pushing them** to a specific target.

In the world of robotics, most robots are taught to "pick and place"—they grab an object to move it. This project focuses on **non-prehensile manipulation**, which is just a fancy way of saying **"moving things without grabbing them"**. Think of how you might nudge a heavy couch across the floor because it’s too heavy to lift; this code gives a robot that same ability to solve tasks using only nudges and slides.

## 🧠 How the Robot "Thinks"
For a robot to push an object successfully, it needs to understand physics and plan ahead. This project uses three main ideas to make that happen:

### 1. Thinking Ahead (MPC)
The robot doesn't just push blindly. It constantly **"imagines" the future** by simulating different paths. It looks at where the object is, tries out different nudges in its mind, and chooses the one that gets the object closest to the goal.

### 2. Automatic Touching (Contact-Implicit)
Usually, human programmers have to tell a robot exactly when and where to touch an object. Our system is smarter—it **automatically figures out the best time to make contact** and the best time to let go as part of its own internal math.

### 3. Exploring the Table (Sampling)
Sometimes, a robot can get "stuck" trying to push an object (like trying to move a 'T' shape around a corner). To fix this, our robot **"scouts" the area** by checking many different positions for its hand. It then picks the one that makes the physics of the push easiest to solve.

## ⚡ The Math Trick: ADMM
Calculating how an object slides across a table (considering friction, weight, and collisions) is incredibly difficult and usually takes a computer a long time. 

To make the robot move in **real-time**, we use a mathematical method called **ADMM**. Instead of trying to solve one giant, impossible physics puzzle all at once, ADMM breaks the puzzle into **small, easy pieces** and solves them one by one very quickly. This allows the robot to react to what it sees almost instantly.

***

# Project Component Overview: Push Anything ADMM

This repository implements a **sample-based Contact-Implicit Model Predictive Control (CI-MPC)** framework, specifically **C3+**, using **PyDrake** for non-prehensile robotic manipulation. 

### 📂 Root Directory
*   **`main.py`**: The primary entry point. It orchestrates the simulation loop, initializes the PyDrake hardware station, and manages real-time visualization through Meshcat.
*   **`setup.sh`**: A shell script for environment initialization and dependency installation.
*   **`environment.yml`**: The Conda configuration file defining dependencies like PyDrake and associated robotics libraries.

### 🧠 `control/` (Controller Logic)
This directory contains the mathematical core of the project, focusing on trajectory optimization through contact.
*   **MPC Implementation (C3/C3+)**: Implements the "Consensus Complementarity Control Plus" algorithm. It uses an MPPI-style sampling architecture to evaluate multiple potential future trajectories in parallel.
*   **Custom ADMM Solver**: A specialized Alternating Direction Method of Multipliers solver that computes rigid body dynamics through frictional contacts. It alternates between solving a smooth Quadratic Program (QP) and a non-smooth proximal projection to enforce non-penetration and friction cone constraints.
*   **LCS Extraction**: Logic to dynamically extract the system's mass matrix, Coriolis terms, and **Linear Complementarity System (LCS)** contact Jacobians directly from the Drake simulation context.

### 🌍 `sim/` & `models/`
*   **`sim/`**: Defines the 3D simulation environment, including the 7-DOF Franka Panda arm, the table, and manipulable rigid bodies.
*   **`models/`**: Stores URDF and 3D mesh files for the robot and various objects.

***

**💡 Sidenote: What I am doing / What I am going to do**

*   **Current Progress:** I am implementing a real-time, sample-based CI-MPC (C3+) in PyDrake. This involves bypassing the exponential complexity of standard mixed-integer programs by using a parallelized ADMM-based solver to "hallucinate" contact sequences.
*   **Future Roadmap:** My next steps involve extending this pipeline to **3D non-prehensile manipulation** and integrating **online model learning** or **ContactSDF** representations to improve efficiency and adapt to unknown object properties.

***

### 📚 References

 **Push Anything**: H. Bui et al., "Push Anything: Single- and Multi-Object Pushing From First Sight with Contact-Implicit MPC," *arXiv preprint arXiv:2510.19974v2*, 2025.

 **Consensus Complementarity Control (C3)**: A. Aydinoglu et al., "Consensus Complementarity Control for Multi-Contact MPC," *IEEE Transactions on Robotics*, vol. 40, pp. 3879–3896, 2024.

 **ContactSDF**: W. Yang and W. Jin, "ContactSDF: Signed Distance Functions as Multi-Contact Models for Dexterous Manipulation," *arXiv preprint arXiv:2408.09612v2*, 2024.

 **LCS**: W. Heemels, J. M. Schumacher, and S. Weiland, "Linear complementarity systems," *SIAM Journal on Applied Mathematics*, vol. 60, no. 4, pp. 1234–1269, 2000.
```