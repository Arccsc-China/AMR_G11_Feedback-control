# **Tello Drone Feedback Control Simulation**

This repository contains the simulation module for implementing feedback control of a Tello drone, developed as part of **Coursework 3** in **AERO60492 – Autonomous Mobile Robots** at **The University of Manchester**.

## **Project Objective**

The aim of this project is to design and implement a simple **feedback control algorithm** for position stabilization of an **Unmanned Aerial System (UAS)**. This involves processing sensor measurements (e.g., GPS, attitude, and velocity) to generate control commands that guide the drone to a desired 3D position and yaw orientation.

Fundamental to robotics, this capability enables more complex guidance, navigation, and decision-making tasks. The controller's role is to produce desired velocity commands (`x`, `y`, `z`) and yaw rate based on the current UAV state.

---

## **Simulation Environment**

* The simulation uses a **custom PyBullet-based 3D environment**.
* The quadcopter model flies over an empty flat terrain.
* For `controller.py`, input data includes:

  * UAV current position and orientation (`x`, `y`, `z`,`yaw`)
  * UAV target position and orientation (`target_x`, `target_y`, `target_z`,`target_yaw`)
* Output:

  * Velocity commands and yaw rate (`vel_x`, `vel_y`, `vel_z`,`yaw_rate`)
* The provided parameters are tuned for simulation use and **may require scaling** for real-world drone applications.

---

## **Installation & Requirements**

### Prerequisites

* Python 3.11 (recommended)

### Required Packages

Install dependencies via pip:

```bash
pip install pybullet numpy matplotlib
```

---

## **Running the Simulation**

### Step-by-step

1. Clone or download this repository.
2. Create a Python environment and install the required packages.
3. Choose the controller implementation (`PID` or `LQR`).
4. Select one of the launch scripts:

#### Standard Run:

```bash
python run.py
```

#### Debug Run (includes convergence plots):

```bash
python run_debug.py
```

Alternatively, run via your IDE after selecting the correct interpreter.

---

## **Target Setup**

Target positions are read from a CSV file:

* File: `targets.csv` (same directory as `run.py`)
* Format:

  ```
  target_x   target_y   target_z   target_yaw
  ```

---

## **Simulator Key Controls**


| Key  | Function                           |
| ---- | ---------------------------------- |
| `r`  | Reset UAV position and orientation |
| `→` | Advance to next target in the list |
| `←` | Return to previous target          |
| `q`  | Quit the simulation                |

---

## **Contributors**

* **Shuyan Zhang** – Documentation, LQR Controller
* **YuChuan Liao** – PID Controller

---

## **Course Context**

This simulation was developed as part of the **Autonomous Mobile Robots (AERO60492)** module, Spring 2025 session, under the School of Engineering, University of Manchester.
