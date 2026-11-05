# Affective Embodied AI: Emotion-Driven Physical Simulation & Urgency-Aware Navigation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Physics](https://img.shields.io/badge/Dynamics-Lagrangian-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

## Overview

**Affective Embodied AI** is a full-stack, dual-pipeline intelligent system that bridges the gap between **Cognitive NLP (Affective Computing)** and **Physical Execution (Multi-body Dynamics)**. 

Unlike traditional robotic simulations that rely on purely kinematic playback, this project integrates a pure Lagrangian physics engine with a learned behavioral cloning network (DNN) and a PPO-guided sampling planner. It demonstrates how abstract textual events (e.g., *"A dog is chasing me!"*) are translated into high-dimensional emotional states (Valence & Urgency) to dynamically modulate the **physical torque**, **locomotion speed**, and **obstacle avoidance strategies** in real time.

> **[ScreenRecording2026-11-05at18.29.08-ezgif.com-video-to-gif-converter.gif]** *(Show the 4-panel UI running: Navigation, Arm, Leg, Affective Monitor)*

---

## System Architecture

The project is decoupled into four heterogeneous modules, unified by a central Affective State Machine (`main.py`):

### 1. Affective Perception (NLP Layer)
- **Amygdala Layer (ValenceNet):** Utilizes a fine-tuned `SentenceTransformer (MiniLM)` to predict emotional valence (-1.0 to 1.0) from text events.
- **Urgency Policy (UrgencyNet):** A lightweight regression network trained via MSE Loss to output the urgency score (0.0 to 1.0).
- **Cognitive Modulation:** Valence directly scales target trajectories and feed-forward torques. Urgency scales the navigation planner's maximum speed.

### 2. TM (Target-Motivated) Navigation
- **Safety Layer:** Monte Carlo sampling generates hundreds of candidate forces, filtered by short-horizon (15-step) forward physics rollouts to eliminate collisions.
- **Optimal Layer:** A learned PPO Critic \(V(s)\) evaluates the long-term value of the terminal state. 
- **Objective Function:** `Score = V(s_terminal) + ΔDistance * (1 + Urgency) - Energy_Penalty`

### 3. Arm Dynamics (8-DOF Analytical Engine)
- **Pure Lagrangian Dynamics:** Solved via full \(M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) = \tau\) with explicit Coriolis and gravity matrices.
- **Forward/Inverse Kinematics:** Implements full 3D Jacobian matrices and real-time RK4 numerical integration.
- **Emotion Injection:** Target joint angles \(q\) and feed-forward torques \(\tau\) are dynamically amplified based on the Affective Valence.

### 4. Leg Dynamics (2-DOF Planar Engine + DNN)
- **Behavior Cloning:** A DNN (`TorqueNet`) trained on **4.25 million** data points learns the mapping from 20-dim biomechanical features to 4-dim optimal joint torques, effectively acting as "muscle memory" for real-time inference.
- **High-Dynamic Tracking:** Achieves perfect tracking of extreme movements (e.g., high-knee running) by combining DNN feed-forward predictions with a PD feedback controller under severe physical constraints.

---

## Mathematical Foundations

**1. Lagrangian Rigid-Body Dynamics**
\[
 \tau = M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) 
\]
Where \(M(q)\) is the mass matrix, \(C\) is the Coriolis/centrifugal term, and \(G\) is the gravity vector. 

**2. Affective Modulation**
\[
 \tau_{final} = \tau_{optimum} \times (1 + \alpha \cdot \text{Valence}) 
\]
Torque and trajectory gains are modulated by the Amygdala network output, simulating human adrenaline release during high-arousal states.

---

## Directory Structure

```text
Physical_Intelligence/
├── amygdala_layer/              # NLP text-to-emotion model (Valence)
├── urgency_policy/              # NLP text-to-urgency regression model
├── tm_algorithm/                # PPO-guided Monte Carlo sampling planner
├── video_2_simulation_4_arm/    # 8-DOF pure Lagrangian physics engine & pipeline
├── video_2_simulation_4_leg/    # 2-DOF physics engine + DNN torque predictor
├── main.py                      # Unified 4-panel real-time visualization dashboard
└── README.md
