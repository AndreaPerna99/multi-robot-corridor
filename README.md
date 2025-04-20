# Aggregative Optimization for Multi-Robot Systems

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)  
Distributed multi-robot coordination using Aggregative Tracking, Potential Functions, and ROS 2-based optimization in real time.

---

## 🧠 About the Project

This project explores decentralized control and optimization strategies for **multi-robot systems** using **Aggregative Tracking**, **Potential Functions**, and a **ROS 2-based distributed implementation**.

It represents the second part of the final project for the **Distributed Autonomous Systems** course at the MSc in Automation Engineering, University of Bologna.

The project is structured into three main tasks:

- 🛰 **Task 2.1** – Multi-robot surveillance using Aggregative Tracking (Python)
- 🤖 **Task 2.2** – Real-time distributed implementation in ROS 2
- 🚧 **Task 2.3** – Corridor navigation with Potential Functions and Projected Optimization (Python)

---

## 🗂 Project Structure

```
📦 aggregative-multirobot/
├── task2_1_ws/                        # ROS 2 workspace for Task 2.1
├── task2_3_ws/                        # ROS 2 workspace for Task 2.3
├── Images/, Videos/                  # Plots and 2D/3D animations
├── Task2_1.py                         # Python simulation - surveillance
├── Task2_3.py                         # Python simulation - corridor
├── Projected_Aggregative_Tracking.py # Projected optimization implementation
├── Project_Functions.py              # Shared utilities
├── README.md
```

---

## ⚙️ How to Run

### 🔹 Task 2.1 – Multi-Robot Surveillance via Aggregative Tracking (Python)

Simulates robots jointly tracking local targets and a global goal, while maintaining a formation using an aggregative optimization strategy.

```bash
$ python3 Task2_1.py
```

🔧 **Adjustable Parameters:**
- `NN`, `MAXITERS`, `step-size`, `radius`, `b`, etc.
- Local target motion options (enable with `moving_loc_targets=True`)
- Aggregation weights: `gamma_r_lt`, `gamma_agg`, `gamma_bar`

📈 **Outputs:**
- Cost plots, gradient evolution, tracking error
- 2D animation of agent and target trajectories

---

### 🔹 Task 2.2 – Real-Time ROS 2 Implementation

Full distributed execution using ROS 2 nodes and launch files. Agents communicate in real time, with visualizations via RVIZ and a centralized animation tool.

#### ▶ Task 2.2.1 – Surveillance in ROS 2

```bash
$ cd task2_1_ws/
$ colcon build --symlink-install
$ . install/setup.bash
$ ros2 launch task2_ros task2_ros.launch.py
```

Then visualize:

```bash
$ cd src/task2_ros/centralized_animation/
$ python3 centralized_animation.py
```

---

#### ▶ Task 2.2.3 – Corridor Navigation in ROS 2

```bash
$ cd task2_3_ws/
$ colcon build --symlink-install
$ . install/setup.bash
$ ros2 launch task2_3 task2_3.launch.py
```

Then:

```bash
$ cd src/task2_3/centralized_animation/
$ python3 centralized_animation.py
```

---

### 🔹 Task 2.3 – Corridor Navigation (Python)

Agents must cross a corridor while avoiding walls and reaching targets. Two methods are provided:

#### ➤ Option 1: Potential Functions

```bash
$ python3 Task2_3.py
```

🧩 Parameters include corridor dimensions, wall avoidance (`avoid_walls=True`), gain tuning, and layout options (`random_init` or circular start).

#### ➤ Option 2: Projected Aggregative Tracking

```bash
$ python3 Projected_Aggregative_Tracking.py
```

👁‍🗨 Based on a 3-stage approach (to-corridor, through-corridor, to-targets) using projections to stay within feasible bounds.

📈 **Outputs:**
- Cost and gradient evolution
- Tracking performance
- 2D animation of agent paths (can be slowed down due to obstacle logic)

---

## 📊 Output & Evaluation

- 📈 Cost and gradient convergence
- 📌 Barycenter tracking error
- 🎯 Trajectory and formation consistency
- 🧠 Real-time vs centralized performance comparison

---

## 👨‍🎓 Authors

Group 3 – MSc Automation Engineering, University of Bologna  
- Andrea Perna  
- Gianluca Di Mauro  
- Meisam Tavakoli  

📧 andrea.perna3@studio.unibo.it

---

## 👩‍🏫 Supervisors

- Prof. Giuseppe Notarstefano  
- Prof. Ivano Notarnicola  
- Dr. Lorenzo Pichierri

---

## 📎 Resources

- [📘 Full Report (PDF)](./report_group_03.pdf) *(see pages 19–33)*  
- [📄 Notebook Output (PDF)](./Task2_Output.pdf)  
- 🎞️ Animations and visualizations in `/Videos/`

---

## 📜 License

All rights reserved. Educational use only.
