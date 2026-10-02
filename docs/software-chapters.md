# Software chapter map

This is a section-by-section map of `workspace/software/`. Each chapter has a different purpose; the balance environment and shared PPO trainer are detailed in [Balancing bot training](balancing-bot-training.md) and [PPO trainer reference](ppo-trainer.md).

## 01 — Test the model in MuJoCo

`workspace/software/01-test-model-in-mujoco/` contains a Python PID example and model motion inspection notebook/script. Use it to verify the simulated mechanism and basic control before training a learned policy. It is a classical-control baseline and model exploration stage, not PPO training.

## 02 — Balance bot with PPO

`workspace/software/02-balance-bot-with-ppo/` defines `BalanceBotEnv`, checks the environment in scalar and vector forms, and provides baseline and curriculum PPO notebooks. The environment exposes four sensor-derived values and accepts two normalized wheel torques. The curriculum starts by rewarding balance and then adds position/yaw shaping while continuing the learned agent. See the dedicated training guide for the exact formulas and procedure.

## 03 — Simulation to real

`workspace/software/03-sim-to-real/` contains Python ONNX inference checks, the main Arduino balance-bot firmware, a tuned variant, sensor/IMU calibration tests, and a static inference test. The Python example runs a single policy inference. Firmware reads sensors, prepares the observation, executes generated C weights, and drives motors. The simulator and hardware implementations must agree on observation order, sign, units, action scaling, and model weights. Successful inference is not evidence of safe physical behavior.

## 04 — Domain randomization

`workspace/software/04-domain-randomization/` adds `BalanceBotEnv` variations and a staged PPO notebook. The documented phases start with balance, add position/yaw penalties, then introduce observation noise/action delay, motor noise/pushes, mass/friction variation, motor-gain variation, and random axle torques. Consult `balance_bot_env_dr.py` and the notebook's `DomainRandomConfig` values for the precise settings. Randomization exposes the policy to variation in simulation; it does not establish sim-to-real robustness without measured evaluation.

## 05 — Command conditioning

`workspace/software/05-command-conditioning/` adds command-conditioned training and tests. `balance_bot_env_cmd.py` defines the command-aware environment; the training notebook progresses from balance and position/yaw penalties to small velocity commands, yaw commands, mid-episode command sampling, then the same kinds of observation/action/physics randomization. Its later stages add observation noise/action delay, motor noise/pushes, mass/friction variation, motor-gain variation, and axle torque disturbances. `pid_command_test.ipynb` and `vector_test.ipynb` explore controller and vector-environment behavior. The policy must respond to command inputs while balancing. Use the environment and notebook source for the current observation layout, command ranges, and reward coefficients.

## 06 — Remote control

`workspace/software/06-remote-control/` contains the remote-control firmware and thumbstick test material. It concerns hardware input/control integration rather than training the PPO policy. Check the individual sketch's wiring, board, and library assumptions before compiling or connecting hardware.

## Shared reinforcement-learning code

`workspace/software/rl/ppo_trainer.py` provides `PPOConfig`, actor/critic networks, `train`, `evaluate`, checkpoint handling, TensorBoard CSV export, and actor ONNX export. `onnx_actor_to_c.py` converts the actor graph's linear/tanh operations and weights into a C header for embedded inference. The shared trainer API and algorithm are documented in [PPO trainer reference](ppo-trainer.md).
