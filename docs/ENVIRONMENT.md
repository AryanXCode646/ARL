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
