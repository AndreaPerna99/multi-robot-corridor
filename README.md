# 🧠 Aggregative Optimization for Multi-Robot Systems

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)  
Distributed multi-robot coordination using Aggregative Tracking, Potential Functions, and ROS 2-based optimization in real-time.

---

## 📚 About the Project

This project implements a fully distributed control framework for **multi-robot systems**, relying on **aggregative tracking algorithms** to maintain formation while reaching individual and global targets. It features:

- 🛰 **Task 2.1** – Multi-robot surveillance and goal-tracking via aggregative tracking.
- 🚧 **Task 2.3** – Navigation through a constrained corridor using potential functions and projected optimization.
- 🤖 **Task 2.2** – Real-time ROS 2 implementation of both surveillance and navigation in distributed robotic systems.

The simulation is developed in Python for central analysis and extended to **ROS 2** for real-time execution. This project is the second part of the final exam project for the Distributed Autonomous System course at the Master Degree in Automation Engineering, University of Bologna.

---

## 🗂 Project Structure

```
📦 aggregative-multirobot/
├ 📁 task2_1_ws/                      # ROS 2 workspace for Task 2.1
├ 📁 task2_3_ws/                      # ROS 2 workspace for Task 2.3
├ 📁 Images/, Videos/                # Plots and 2D/3D animations
├ 📄 Task2_1.py                       # Python simulation for surveillance
├ 📄 Task2_3.py                       # Python simulation for corridor navigation
├ 📄 Projected_Aggregative_Tracking.py
├ 📄 Project_Functions.py            # Shared utility functions
├ 📄 README.md
```

---

## ⚙️ How to Run

### 🔹 Task 2.1 – Surveillance with Aggregative Tracking

> Simulates multiple robots pursuing local and global targets while maintaining formation.

```bash
$ python3 Task2_1.py
```

- Set parameters inside the script (`NN`, `MAXITERS`, local/global goals, motion options).
- Choose between static or moving local targets.
- Outputs: cost plots, trajectory animations, and error metrics.

---

### 🔹 Task 2.3 – Corridor Navigation

> Robots traverse a corridor avoiding walls via:
- ⚠️ **Potential Functions**
- 📐 **Projected Aggregative Tracking**

```bash
$ python3 Task2_3.py
# or
$ python3 Projected_Aggregative_Tracking.py
```

- Use `avoid_walls=True` to enable obstacle avoidance.
- Modify `corr_width`, `corr_length`, and `barycenter_goal_coordinates`.
- Outputs: agent paths, barrier effects, and cost/gradient plots.

---

### 🔹 Task 2.2 – ROS 2 Real-Time Implementation

> Brings both strategies to real-time execution using **ROS 2 nodes**, **launch files**, and **RVIZ**.

#### ▶ Task 2.2.1 – Surveillance

```bash
$ cd task2_1_ws/
$ colcon build --symlink-install
$ . install/setup.bash
$ ros2 launch task2_ros task2_ros.launch.py
```

Then:

```bash
$ cd src/task2_ros/centralized_animation/
$ python3 centralized_animation.py
```

#### ▶ Task 2.2.3 – Corridor Navigation

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

## 📊 Output & Evaluation

- 📈 Cost and gradient evolution plots
- 📌 Position tracking errors and convergence metrics
- 📺 2D & 3D animated trajectories
- 🧠 Comparison between centralized and distributed implementations

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

- [📘 Final Report (PDF)](./report_group_03.pdf) *(pages 19–33 relevant)*  
- [📄 Code output and animations included in Videos/*]*  

---

## 📜 License

All rights reserved. Educational use only.
