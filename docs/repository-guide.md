# Repository guide

## Purpose

The repository supports a reinforcement-learning-for-robotics video series. It contains a Docker-based development desktop, a MuJoCo model and learning experiments for a two-wheel balancing robot, and later simulation-to-real and robot-control material. The top-level README is the operational reference for Docker and native macOS setup; this guide explains how the pieces fit together.

## Top-level layout

| Path | Role |
| --- | --- |
| `Dockerfile.cpu`, `Dockerfile.macos`, `Dockerfile.bare` | Container variants for the development environment. |
| `scripts/` | Desktop entrypoint and launchers/configuration used by the container. |
| `workspace/mechanical/FreeCAD/bala2-fire/` | FreeCAD assets, simplified MJCF/URDF models, meshes, and model-generation scripts. |
| `workspace/software/01-test-model-in-mujoco/` | Model inspection and simple PID/control experiments. |
| `workspace/software/02-balance-bot-with-ppo/` | Gymnasium environment and PPO training/evaluation notebooks. |
| `workspace/software/03-sim-to-real/` | ONNX inference checks and embedded balance-bot firmware examples. |
| `workspace/software/04-domain-randomization/` | Environment randomization extensions and staged training. |
| `workspace/software/05-command-conditioning/` | Command-conditioned velocity/yaw control experiments. |
| `workspace/software/06-remote-control/` | Remote-control sketches and input-device tests. |
| `workspace/software/rl/` | Shared PPO trainer and ONNX-to-C conversion utility. |

## Runtime choices

The root README describes building/running the Docker environment, native macOS setup, and the MPS option for policy-network computation. In Docker on macOS, the Linux process cannot use the Apple GPU; simulation runs on CPU. In native macOS training, `PPOConfig.cuda` selects CUDA when available or MPS on supported PyTorch builds, while MuJoCo simulation remains on CPU. Device transfers may reduce the benefit for the small policy network. Rendering under native macOS requires launching the MuJoCo viewer through `mjpython`; setting `MUJOCO_RENDER=0` disables notebook rendering.

The repository root README is the source for dependency installation and launch commands. The balance-bot chapter README gives folder-specific notebook guidance.

## Recommended learning path

1. Inspect the simplified MJCF robot and run the model/PID examples in chapter 01.
2. Read [Balancing bot training](balancing-bot-training.md), then run the environment checks and PPO notebook in chapter 02.
3. Use the curriculum variant to continue training the same policy under additional reward terms.
4. Read chapter 03 before exporting or testing a model for embedded inference.
5. Explore domain randomization and command conditioning after understanding the base observation, action, reward, and PPO loop.

## Source-of-truth and generated artifacts

Python files define environment, trainer, and converter behavior. Notebook code cells show how those APIs are used; notebook outputs are historical snapshots and can become stale as source code changes. Model checkpoints, TensorBoard event files, CSV exports, ONNX files, and generated C headers are run artifacts, not source definitions. Keep a deployed `actor.h` matched to the ONNX actor from which it was generated.

The root `.gitignore` excludes generated and local development files. Large CAD assets and notebook outputs may be tracked separately from the small Python sources; inspect Git status before committing so local changes are not accidentally included.
