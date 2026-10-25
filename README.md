# Embodied Intelligence: Affect-Driven 3D Robotic Simulation

> Integrating Inverse/Forward Dynamics, model-based amygdala, and a new navigation named TM navigation with urgency policy for 3D arm and leg simulations based on double pendulum model.

---

## Key Innovations

- **Dual-Pipeline System**: Separates physical execution (Simulation Core) from spatial decision-making (TM Navigation).
- **Cognitive Modulation**: Employs an Amygdala network to modulate joint torques based on emotional valence, and an Urgency network to dynamically scale locomotion speed.
- **Custom TM Algorithm**: The optimal trajectory is selected using a multi-objective evaluation function.

$$
\text{Score}(a_i) = Q_\phi(s_t, a_i) - \lambda_{\text{energy}} \|a_i\|^2 + u_t \cdot \Delta D(a_i)
$$

---

## System Architecture

The framework is divided into two core pipelines:

### Part 1: Simulation Core (Inverse & Forward Dynamics)

This pipeline translates video data into physics-driven emotional animations.

1. **Inverse Dynamics**: Extracts skeletal kinematics from video datasets, calculates the required torques, and extracts the **Optimum $\tau$** for standard baseline movements.
2. **Affect Modulation (Amygdala)**: The event text is processed to generate an emotional valence score. This score dynamically scales the Optimum $\tau$ to generate the **Final $\tau$**, dictating the final swing amplitude.
3. **Forward Dynamics**: The Final $\tau$ is fed into the 3D rigid-body physics engine (using RK4 numerical integration, mass matrices, and gravity terms) to render the final physical animation.

### Part 2: TM (Target-Motivated) Navigation (Obstacle Avoidance)

1. **Urgency Policy (The 3rd Term)**: A lightweight NLP pipeline reads environmental events (e.g., "A dog is chasing me!"). A Sigmoid-activated neural network outputs an Urgency score ($u_t$). This score directly scales the distance bonus.
2. **PPO Critic (The 1st & 2nd Terms)**: A trained PPO Critic evaluates the long-term spatial value ($Q_\phi$) of the trajectory, while an energy penalty ($\lambda$) prevents erratic movements.
3. **TM Evaluation**: Hundreds of physical rollouts are generated per frame. Collisions are filtered out (Safety Layer). The remaining trajectories are scored using our custom formula.

---

## Mathematical Foundations

### 1. Dynamics & Emotional Modulation

**Baseline Torque (Optimum $\tau$):**

$$
\tau_{\text{optimum}} = M(q)\ddot{q} + G(q) 
$$

**Final Torque (With Emotion):**

$$
\tau_{\text{final}} = \tau_{\text{optimum}} \times (1 + \alpha \cdot \text{Valence}) 
$$

### 2. The TM Algorithm Evaluation Function

For each safe trajectory $a_i$, the optimal action is selected by maximizing the following score:

$$
\text{Score}(a_i) = Q_\phi(s_t, a_i) - \lambda_{\text{energy}} \|a_i\|^2 + u_t \cdot \Delta D(a_i)
$$

- $Q_\phi(s_t, a_i)$: Long-term value evaluated by the trained PPO Critic.
- $\lambda_{\text{energy}} \|a_i\|^2$: Energy penalty to prevent erratic movements.
- $u_t \cdot \Delta D(a_i)$: Dynamic Urgency bonus, where $u_t$ is the Sigmoid output (0~1) and $\Delta D$ is the distance reduction to the target.
