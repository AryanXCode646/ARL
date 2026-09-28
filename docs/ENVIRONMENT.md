# Simulated 3D Drone Environment Specification

## Mathematical Model

The drone is modeled as a 3-DOF kinematic point mass subject to commanded thrust and linear aerodynamic drag damping.

### State Equations

Given time step $\\Delta t = 0.1\\text{ s}$:

$$\\mathbf{a}_t = \\text{clip}(\\mathbf{u}_t, -1, 1) \\cdot a_{\\max}$$

$$\\mathbf{v}_{t+1} = \\mathbf{v}_t (1 - c_{\\text{drag}} \\Delta t) + \\mathbf{a}_t \\Delta t$$

$$\\mathbf{v}_{t+1} = \\text{clip}(\\mathbf{v}_{t+1}, -v_{\\max}, v_{\\max})$$

$$\\mathbf{p}_{t+1} = \\mathbf{p}_t + \\mathbf{v}_{t+1} \\Delta t$$

Where:
- $a_{\\max} = 3.0\\text{ m/s}^2$: Maximum linear acceleration
- $v_{\\max} = 5.0\\text{ m/s}$: Maximum linear velocity
- $c_{\\text{drag}} = 0.25$: Aerodynamic linear drag coefficient

---

## Sensor Model (16-Ray LiDAR)

The simulated drone carries a 16-ray spherical rangefinder:
- 8 rays in the horizontal plane ($0^\\circ, 45^\\circ, \\dots, 315^\\circ$)
- 4 rays elevated at $+45^\\circ$ ($0^\\circ, 90^\\circ, 180^\\circ, 270^\\circ$)
- 4 rays inclined at $-45^\\circ$ ($0^\\circ, 90^\\circ, 180^\\circ, 270^\\circ$)

Ray-sphere intersection tests are computed analytically against each spherical obstacle in the arena up to a maximum range of $15.0\\text{ m}$. Readings are normalized to $[0.0, 1.0]$ where $0.0$ indicates an immediate obstacle and $1.0$ indicates free space.

---

## Reward Formulation

At each step $t$:

$$R_t = w_{\\text{progress}} (d_{t-1} - d_t) - c_{\\text{step}} - c_{\\text{action}} \\|\\mathbf{u}_t\\|^2 + R_{\\text{terminal}}$$

Where:
- $d_t = \\|\\mathbf{p}_t - \\mathbf{g}\\|$: Euclidean distance to target
- $w_{\\text{progress}} = 2.0$: Positive reward for moving toward the target
- $c_{\\text{step}} = 0.05$: Time penalty per step
- $c_{\\text{action}} = 0.01$: Smoothness penalty on control effort
- $R_{\\text{terminal}}$:
  - $+100.0$ if reached target ($d_t \\le 1.0\\text{ m}$)
  - $-50.0$ if collided with obstacle or arena boundary

---

## Trajectory-Quality & Safety Evaluation Metrics

The evaluation engine computes deterministic trajectory-quality and safety metrics across evaluation rollouts:

### Trajectory-Quality Metrics

- **Path length ($L$)** [$\text{m}$]: Cumulative Euclidean distance traveled across trajectory waypoints:
  $$L = \sum_{t=0}^{T-1} \|\mathbf{p}_{t+1} - \mathbf{p}_t\|_2$$
- **Straight-line distance ($D_0$)** [$\text{m}$]: Euclidean distance from initial drone position to target:
  $$D_0 = \|\mathbf{g} - \mathbf{p}_0\|_2$$
- **Path efficiency ($\eta$)** [dimensionless]: Ratio of straight-line distance to actual path length, clamped to $[0.0, 1.0]$:
  $$\eta = \begin{cases} \text{clip}\left(\frac{D_0}{L}, 0.0, 1.0\right) & \text{if } L > 0 \\ 0.0 & \text{if } L \le 0 \end{cases}$$

### Safety & Dynamics Metrics

- **Minimum obstacle clearance ($d_{\min}$)** [$\text{m}$]: Closest distance from drone position to any spherical obstacle surface over the rollout:
  $$d_{\min} = \min_{t, i} \left(\|\mathbf{p}_t - \mathbf{c}_i\|_2 - r_i\right)$$
  *(Measured to obstacle surface, where $\mathbf{c}_i$ is the center and $r_i$ is the radius. Negative values indicate penetration.)*
- **Maximum velocity ($v_{\max}$)** [$\text{m/s}$]: Peak instantaneous linear speed attained during rollout:
  $$v_{\max} = \max_t \|\mathbf{v}_t\|_2$$
- **Maximum acceleration ($a_{\max}$)** [$\text{m/s}^2$]: Peak instantaneous linear acceleration magnitude:
  $$a_{\max} = \max_t \|\mathbf{a}_t\|_2$$
- **Obstacle collisions**: Count and rate of terminal impacts against spherical obstacle surfaces.
- **Boundary collisions**: Count and rate of terminal impacts against arena bounding planes ($X, Y, Z$).
