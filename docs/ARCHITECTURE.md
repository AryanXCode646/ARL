# AdaptiveRL Architecture Specification

## 1. System Overview

**AdaptiveRL** is a modular reinforcement learning framework engineered to train, evaluate, and benchmark agents across multiple diverse environments. Rather than tightly coupling agent algorithms to a specific task, AdaptiveRL separates the problem into clear, orthogonal layers interacting through standardized interfaces.

```
                         +-----------------------------------+
                         |       Configuration Layer         |
                         |   (YAML Schemas & Seed Manager)   |
                         +-----------------+-----------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
        +-----------------------+                     +-----------------------+
        |  Environment Layer    |                     |    Algorithm Layer    |
        |  - Gymnasium API      |                     |  - BaseAlgorithm      |
        |  - Registry & Factory |                     |  - SB3 Wrappers (PPO) |
        +-----------+-----------+                     +-----------+-----------+
                    |                                             |
                    +----------------------+----------------------+
                                           |
                                           v
                         +-----------------------------------+
                         |          Training Engine          |
                         |  (Trainer, Callbacks, Checkpoints)|
                         +-----------------+-----------------+
                                           |
                                           v
                         +-----------------------------------+
                         |         Evaluation Engine         |
                         |  (Metrics, Scenarios, Benchmarks) |
                         +-----------------+-----------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
        +-----------------------+                     +-----------------------+
        |   Model Management    |                     | Visualization Engine  |
        | (Artifacts & Metadata)|                     |  (Plots & Renderers)  |
        +-----------------------+                     +-----------------------+
```

---

## 2. Core Architectural Decisions

### 2.1 Why the `src/` Layout?
AdaptiveRL utilizes the `src/` layout (`src/adaptive_rl/`) instead of a flat root structure:
* **Import Parity:** Prevents tests and scripts from accidentally importing the raw working directory instead of the installed package.
* **Packaging Reliability:** Ensures editable installs (`pip install -e .`) and distributed wheel builds reflect the true installed artifact.
* **Tooling Compatibility:** Cleanly isolates tooling caches (`.pytest_cache`, `.mypy_cache`, `.ruff_cache`) and prevents namespace pollution.

### 2.2 Separation Between Environments and Algorithms
Algorithms never import or depend directly on concrete environments.
* **Gymnasium Contract:** All environments conform strictly to Farama Gymnasium semantics (`observation_space`, `action_space`, `reset(seed=...)`, `step(action)`).
* **Algorithm Invariance:** The RL algorithm operates purely on abstract tensor or array spaces without knowing whether the domain is a 2D GridWorld, continuous navigation, traffic intersection, or 3D drone simulation.

### 2.3 Environment Registry Concept
The environment registry acts as a dynamic service locator and factory:
* Environments register with unique identifier keys (e.g. `gridworld`, `navigation`, `drone`).
* The training engine requests `make_env("environment_name", config)` dynamically.
* Protects against duplicate registrations and gives helpful diagnostics when an unrecognized environment is requested.

### 2.4 Training and Evaluation Separation
* **Training (`adaptive_rl.training`):** Focuses exclusively on optimizing policy weights, updating value functions, logging progression, and managing periodic checkpoints.
* **Evaluation (`adaptive_rl.evaluation`):** Evaluates policies deterministically over fixed episode sets, computing uncorrupted metrics (success rate, collision rate, average episodic return, episode length). Training and evaluation scenarios remain strictly isolated.

### 2.5 Configuration System
* Declarative YAML configurations define all experiment parameters (algorithm hyperparameters, environment settings, training timesteps, seed, logging paths).
* Configurations are parsed into strongly-typed Pydantic models (`ExperimentConfig`), catching typos, invalid ranges, and missing fields early with actionable error messages.
* Determinism is enforced by setting seeds across Python `random`, NumPy, PyTorch, and Gymnasium spaces simultaneously.

### 2.6 Model Management
* Models are saved with comprehensive metadata (`ModelMetadata`) recording:
  * Algorithm and version
  * Training environment name
  * Total timesteps trained
  * Hyperparameter dictionary
  * Checkpoint timestamp and path
* Prevents attempting to evaluate or deploy models into mismatched action/observation spaces.

### 2.7 Visualization and Decoupled Rendering
* Rendering logic (`BaseRenderer`) is completely decoupled from environment physics.
* Environments can run in headless mode (e.g., in continuous integration) at maximum execution speed without requiring graphics displays or X11/OpenGL servers.
* Plotting utilities (`PlotManager`) generate training curve visualizations from structured logs without touching the active training process.

### 2.8 Experiment Lifecycle Management
* Every experiment run produces a self-contained artifact bundle in `experiments/results/` and `experiments/logs/`.
* The saved configuration file guarantees that any experiment can be reproduced by another researcher with a single CLI command.

### 2.9 Environment Abstraction & Factory Flow (Environment → Registry → Factory → Training Engine)

The relationship between environments and the training engine follows a strict unidirectional dependency flow mediated by the registry and factory:

```
+------------------+         registers         +------------------------+
| Concrete Env     | ----------------------->  |  Environment Registry  |
| (e.g. GridWorld) |                           |  (Name -> Factory Map) |
+------------------+                           +-----------+------------+
                                                           | resolves
                                                           v
+------------------+       instantiates        +------------------------+
| Training Engine  | <------------------------ |  Environment Factory   |
| (Trainer)        |                           |  (make_env / create)   |
+------------------+                           +------------------------+
```

#### Why the Training Engine Must Depend on the Interface Rather Than Concrete Environments:
1. **Zero Domain Coupling:** The training engine (`Trainer`) requires only that the target environment implements `gymnasium.Env` (`reset() -> (obs, info)`, `step(action) -> (obs, reward, terminated, truncated, info)`). It has zero knowledge of grid cells, lidar rays, traffic signals, or quadrotor equations of motion.
2. **Pluggable Architecture:** Any new environment (e.g. `TrafficEnv` or `DroneNavigationEnv`) can be plugged in simply by defining the Gymnasium subclass and registering it with `register("env_name", factory)`. No code in the training engine, algorithm wrappers, or checkpointing routines needs to change.
3. **Seamless Benchmark Portability:** Standard third-party environments (such as Gymnasium's `CartPole-v1`, `Pendulum-v1`, or `BipedalWalker-v3`) can be trained using the exact same CLI command and training engine without wrapping or rewriting them.
4. **Isolated Testability:** Test suites can use lightweight dummy environments (like `DummyTestEnv`) to test the registry, vectorization, and training loops rapidly without incurring heavy simulation overhead.

