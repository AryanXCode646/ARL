"""AdaptiveRL: Autonomous 3D Drone Navigation Demonstration GUI.

A Streamlit-based presentation interface for college professors and evaluators,
visualizing real-time 3D flight trajectories, LiDAR sensor readings, PPO training,
and multi-episode benchmark evaluation.
"""

import time
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
import streamlit as st
from stable_baselines3 import PPO

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.config import AlgorithmConfig, EnvironmentConfig, ExperimentConfig, TrainingConfig
from adaptive_rl.environments.drone import DroneNavigation3DEnv
from adaptive_rl.evaluation.evaluator import Evaluator
from adaptive_rl.gui import (
    build_arena_3d_figure,
    get_available_models,
    run_drone_simulation_episode,
)
from adaptive_rl.training.trainer import PPOTrainer

# Configure Streamlit Page
st.set_page_config(
    page_title="AdaptiveRL — 3D Drone Demonstration",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    # Sidebar: Project Branding & Navigation
    st.sidebar.title("🚁 AdaptiveRL")
    st.sidebar.markdown(
        "**Autonomous 3D Drone Navigation**\n\n"
        "College Demonstration Prototype using Proximal Policy Optimization (PPO)."
    )

    available_models = get_available_models()
    model_names = [m.name for m in available_models]

    tabs = st.tabs(
        [
            "🎮 Live 3D Flight Demo",
            "📈 PPO Training",
            "📊 Benchmark Evaluation",
            "📘 Architecture & Specs",
        ]
    )

    # ==========================================
    # TAB 1: LIVE 3D FLIGHT DEMO
    # ==========================================
    with tabs[0]:
        st.subheader("Simulated 3D Drone Navigation Flight Deck")

        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([2, 1, 1, 1])

        with ctrl_col1:
            selected_model_name = st.selectbox(
                "Select Policy Checkpoint:",
                options=(
                    ["Default Demo Model"] + model_names
                    if model_names
                    else ["Default Demo Model (Random/Untrained)"]
                ),
                index=0,
                help="Choose a trained PPO model checkpoint or use default initialization.",
            )

        with ctrl_col2:
            demo_seed = st.number_input(
                "Environment Seed:",
                min_value=0,
                max_value=999999,
                value=42,
                step=1,
                help="Locks procedural obstacles and initial drone conditions for determinism.",
            )

        with ctrl_col3:
            num_obs = st.slider("Obstacles:", min_value=0, max_value=8, value=4, step=1)

        with ctrl_col4:
            show_lidar_rays = st.checkbox("Show 16-Ray LiDAR", value=True)

        run_col1, run_col2 = st.columns([1, 4])
        with run_col1:
            launch_clicked = st.button("🚀 Launch Flight", type="primary", use_container_width=True)

        # Execute simulation if triggered or not yet in session
        if launch_clicked or "flight_data" not in st.session_state:
            with st.spinner("Executing simulation dynamics..."):
                env = DroneNavigation3DEnv(num_obstacles=num_obs)
                model_to_use: Optional[PPO] = None

                if available_models and selected_model_name in model_names:
                    model_path = next(p for p in available_models if p.name == selected_model_name)
                    try:
                        model_to_use = PPO.load(str(model_path))
                    except Exception as exc:
                        st.warning(f"Could not load checkpoint ({exc}); using random actions.")
                elif available_models:
                    # Load the newest available model
                    try:
                        model_to_use = PPO.load(str(available_models[0]))
                    except Exception:
                        model_to_use = None

                st.session_state["flight_data"] = run_drone_simulation_episode(
                    env=env,
                    model=model_to_use,
                    seed=int(demo_seed),
                )

        flight_data = st.session_state["flight_data"]
        steps = flight_data["steps"]
        total_steps = len(steps) - 1

        # Outcome Banner
        outcome_color = "#38a169" if flight_data["outcome"] == "SUCCESS" else "#e53e3e"
        st.markdown(
            f"""
            <div style="background-color: {outcome_color}22; border-left: 5px solid {outcome_color}; padding: 12px 16px; border-radius: 4px; margin-bottom: 12px;">
                <h4 style="margin: 0; color: {outcome_color};">Outcome: {flight_data["outcome"]}</h4>
                <p style="margin: 4px 0 0 0; color: #cbd5e0; font-size: 14px;">
                    Completed in <b>{flight_data["total_steps"]}</b> timesteps with cumulative reward <b>{flight_data["total_reward"]:.2f}</b> (Seed: {flight_data["seed"]}).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Scrubber Slider for Trajectory Inspection
        scrub_col, metric_box = st.columns([3, 1])
        with scrub_col:
            selected_step = st.slider(
                "Flight Step Playback:",
                min_value=0,
                max_value=total_steps,
                value=total_steps,
                step=1,
                help="Scrub forward or backward in time to inspect drone position, speed, and sensor range.",
            )

        step_info = steps[selected_step]
        cur_pos = step_info["position"]
        cur_vel = step_info["velocity"]
        cur_lidar = step_info["lidar_ranges"]

        # Build Interactive 3D Plotly Arena
        fig = build_arena_3d_figure(
            bounds=flight_data["bounds"],
            obstacles=flight_data["obstacles"],
            target=flight_data["target"],
            start_pos=flight_data["start_pos"],
            trajectory=[s["position"] for s in steps[: selected_step + 1]],
            current_pos=cur_pos,
            current_vel=cur_vel,
            lidar_rays=flight_data["lidar_rays"],
            lidar_ranges=cur_lidar,
            show_lidar=show_lidar_rays,
            title=f"Autonomous 3D Arena — Step {selected_step}/{total_steps} (Distance to Target: {step_info['distance_to_goal']:.2f}m)",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Step Metrics Row
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Altitude (Z)", f"{cur_pos[2]:.2f} m")
        m2.metric("Ground Speed", f"{step_info['speed']:.2f} m/s")
        m3.metric("Range to Goal", f"{step_info['distance_to_goal']:.2f} m")
        m4.metric("Nearest Obstacle", f"{step_info['min_obstacle_distance']:.2f} m")
        m5.metric("Step Reward", f"{step_info['reward']:+.2f}")
        m6.metric("Cumulative Reward", f"{step_info['cumulative_reward']:+.2f}")

    # ==========================================
    # TAB 2: PPO TRAINING
    # ==========================================
    with tabs[1]:
        st.subheader("Train Autonomous Drone Policy (PPO)")
        st.markdown(
            "Execute real reinforcement learning training using **Stable-Baselines3 PPO** "
            "directly on your CPU. The trained policy will immediately become selectable for evaluation and flight demonstration."
        )

        train_col1, train_col2, train_col3 = st.columns(3)
        with train_col1:
            train_steps = st.select_slider(
                "Training Budget (Timesteps):",
                options=[5000, 10000, 25000, 50000],
                value=25000,
                help="25,000 steps requires ~25 seconds on a standard laptop CPU.",
            )
        with train_col2:
            train_lr = st.selectbox(
                "Learning Rate:",
                options=[0.0001, 0.0003, 0.001],
                index=1,
            )
        with train_col3:
            train_seed = st.number_input("Training Seed:", value=42, step=1)

        start_train_btn = st.button("▶ Start PPO Training", type="primary")

        if start_train_btn:
            st.info(f"Starting training run ({train_steps:,} timesteps on CPU)...")
            prog_bar = st.progress(0.0)
            status_text = st.empty()

            try:
                # Build real PPO trainer
                train_cfg = ExperimentConfig(
                    name="drone_ppo_gui",
                    seed=int(train_seed),
                    algorithm=AlgorithmConfig(
                        name="ppo",
                        learning_rate=float(train_lr),
                    ),
                    environment=EnvironmentConfig(
                        name="drone",
                    ),
                    training=TrainingConfig(
                        total_timesteps=int(train_steps),
                    ),
                    output_dir=Path("artifacts"),
                )

                trainer = PPOTrainer(config=train_cfg)
                t0 = time.time()
                result = trainer.fit()
                duration = time.time() - t0

                prog_bar.progress(1.0)
                status_text.text(f"Training Complete! Total time: {duration:.2f}s")
                st.success(
                    f"Successfully saved trained model checkpoint to: `{result.final_model_path}`"
                )

                st.json(
                    {
                        "total_timesteps": result.total_timesteps,
                        "episodes_completed": result.episodes_completed,
                        "duration_seconds": round(duration, 2),
                        "model_path": str(result.final_model_path),
                        "metadata_path": str(result.metadata_path)
                        if result.metadata_path
                        else None,
                    }
                )
            except Exception as e:
                st.error(f"Training failed: {e}")

    # ==========================================
    # TAB 3: BENCHMARK EVALUATION
    # ==========================================
    with tabs[2]:
        st.subheader("Multi-Episode Deterministic Benchmark Evaluation")
        st.markdown(
            "Evaluate trained policy checkpoints across multiple deterministic episodes to measure "
            "empirical **Success Rate**, **Collision Rate**, and **Mean Return**."
        )

        eval_col1, eval_col2, eval_col3 = st.columns([2, 1, 1])
        with eval_col1:
            eval_model_name = st.selectbox(
                "Evaluation Model:",
                options=model_names if model_names else ["No models found in artifacts/models"],
                index=0,
            )
        with eval_col2:
            eval_episodes = st.slider("Episodes:", min_value=5, max_value=50, value=20, step=5)
        with eval_col3:
            run_eval_btn = st.button("📊 Run Evaluation", type="primary", use_container_width=True)

        if run_eval_btn and model_names:
            target_model_path = next(p for p in available_models if p.name == eval_model_name)
            with st.spinner(f"Evaluating policy over {eval_episodes} episodes..."):
                try:
                    eval_env = DroneNavigation3DEnv()
                    eval_algo = PPOAlgorithm.from_pretrained(target_model_path, env=eval_env)
                    evaluator = Evaluator(algorithm=eval_algo, env=eval_env)
                    res = evaluator.evaluate(
                        num_episodes=int(eval_episodes),
                        deterministic=True,
                    )
                    evaluator.export_json(res, Path("artifacts/evaluation.json"))

                    st.markdown("### Benchmark Summary")
                    e1, e2, e3, e4 = st.columns(4)
                    success_pct = (
                        f"{res.success_rate * 100:.1f}%" if res.success_rate is not None else "0.0%"
                    )
                    collision_pct = (
                        f"{res.collision_rate * 100:.1f}%"
                        if res.collision_rate is not None
                        else "0.0%"
                    )
                    e1.metric("Success Rate", success_pct)
                    e2.metric("Collision Rate", collision_pct)
                    e3.metric("Mean Reward", f"{res.mean_reward:.2f} ± {res.std_reward:.2f}")
                    e4.metric("Mean Episode Length", f"{res.mean_episode_length:.1f} steps")

                    # Plot Per-Episode Results
                    ep_rewards = [rec.return_value for rec in evaluator.last_episode_records]
                    if ep_rewards:
                        ep_fig = go.Figure()
                        ep_fig.add_trace(
                            go.Bar(
                                x=list(range(1, len(ep_rewards) + 1)),
                                y=ep_rewards,
                                name="Episode Return",
                                marker_color="#4299e1",
                            )
                        )
                        ep_fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="#111827",
                            plot_bgcolor="#111827",
                            title="Cumulative Reward per Evaluation Episode",
                            xaxis_title="Episode",
                            yaxis_title="Total Reward",
                            margin=dict(l=0, r=0, b=0, t=40),
                        )
                        st.plotly_chart(ep_fig, use_container_width=True)

                    st.info("Evaluation report exported to `artifacts/evaluation.json`")
                except Exception as exc:
                    st.error(f"Evaluation error: {exc}")

    # ==========================================
    # TAB 4: ARCHITECTURE & SYSTEM SPECS
    # ==========================================
    with tabs[3]:
        st.subheader("System Architecture & Mathematical Specifications")

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.markdown(
                r"""
                ### Reinforcement Learning Framework
                - **Algorithm**: Proximal Policy Optimization (PPO) via Stable-Baselines3
                - **Policy Network**: Multi-Layer Perceptron (`MlpPolicy`) with Orthogonal Weights
                - **Action Space**: Continuous 3D thrust acceleration $a \in [-1.0, 1.0]^3$
                - **Observation Space**: 29 continuous dimensions:
                  - Relative target vector $\mathbf{g} - \mathbf{p}$ (3 dims)
                  - Current velocity $\mathbf{v}$ (3 dims)
                  - Drone position $\mathbf{p}$ (3 dims)
                  - Waypoint target coordinate (3 dims)
                  - Euclidean distance to target (1 dim)
                  - 16-ray spherical LiDAR rangefinder (16 dims)
                - **Discount Factor**: $\gamma = 0.99$
                - **GAE Parameter**: $\lambda = 0.95$
                """
            )

        with info_col2:
            st.markdown(
                r"""
                ### Kinematic 3D Simulation Dynamics
                - **Time Step**: $\\Delta t = 0.1\\text{ s}$
                - **Max Acceleration**: $a_{\\max} = 4.0\\text{ m/s}^2$
                - **Max Velocity**: $v_{\\max} = 8.0\\text{ m/s}$
                - **Aerodynamic Linear Drag**: $c_{\\text{drag}} = 0.05$
                - **Motion Integration**:
                  $$\\mathbf{a}_t = \\text{clip}(\\mathbf{u}_t, -1, 1) \\cdot a_{\\max}$$
                  $$\\mathbf{v}_{t+1} = \\mathbf{v}_t (1 - c_{\\text{drag}} \\Delta t) + \\mathbf{a}_t \\Delta t$$
                  $$\\mathbf{p}_{t+1} = \\mathbf{p}_t + \\mathbf{v}_{t+1} \\Delta t$$
                - **Sensor Model**: 16-ray spherical LiDAR rangefinder ($20.0\\text{ m}$ range)
                - **Terminal Constraints**: Bounded flight arena ($30 \\times 30 \\times 15\\text{ m}$)
                """
            )

        st.markdown("---")
        st.markdown(
            """
            ### Educational Scope & Model Assumptions
            - **Point-Mass Translation**: Simulates 3-DOF translation with aerodynamic drag damping rather than 6-DOF rigid body rotor torque.
            - **Raycast Geometry**: Rangefinder detects obstacles analytically via 3D ray-sphere and ray-box intersection without simulated sensor noise.
            - **Designed for Academic Presentation**: Built to run reliably on student laptops without requiring external GPU infrastructure.
            """
        )


if __name__ == "__main__":
    main()
