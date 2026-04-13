readme_content = """# Push Anything: Contact-Implicit MPC with ADMM

This repository contains a Python implementation of the **C3 (Contact-Implicit MPC)** and **C3+ (Push Anything)** algorithms using **PyDrake**. It simulates a 7-DOF Franka Emika Panda robot arm using non-prehensile manipulation (pushing) to move an object to a target location on a table.

The core of this project is a custom **ADMM (Alternating Direction Method of Multipliers) solver** that computes rigid body dynamics through frictional contacts in real-time, allowing the Model Predictive Control (MPC) algorithm to hallucinate and optimize multi-contact interactions without mixed-integer formulations.

## 📝 Features

* **PyDrake Physics Environment:** Full 3D simulation of a Franka Panda arm, planar workspace, and a dynamically manipulable rigid body.
* **LCS Contact Extraction:** Dynamically extracts mass matrices, Coriolis terms, and Linear Complementarity System (LCS) contact Jacobians directly from the Drake context.
* **Custom ADMM Physics Solver:** Solves the contact dynamics by alternating between a smooth QP (for joint accelerations) and a non-smooth proximal projection (for non-penetration and friction cones). Includes Tikhonov regularization for strict mathematical convexity.
* **Sample-Based MPC (C3+):** Uses an MPPI-style sampling architecture to evaluate parallel future trajectories, allowing the robot to discover complex pushing interactions.
* **Real-time Visualization:** Uses Meshcat to render the robot, the table, the object, and a target visualization sphere.

## 📁 Project Structure