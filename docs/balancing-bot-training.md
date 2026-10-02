# Balancing bot training: from simulator state to deployed actor

This guide follows the main implementation across `workspace/mechanical/FreeCAD/bala2-fire/bala2-fire-simplified.xml`, `workspace/software/02-balance-bot-with-ppo/balance_bot_env.py`, the PPO notebooks, `workspace/software/rl/ppo_trainer.py`, and the sim-to-real chapter. It explains what the code trains, how each update is formed, what the curriculum changes, and what model export means.

## 1. Robot model and simulator

The simplified MJCF file defines the simulated robot, sensors, motors, physics, and simulation timestep. `BalanceBotEnv` loads that model with `mujoco.MjModel.from_xml_path` and creates an `MjData` state. It resolves motor actuator IDs by MJCF name and reads named IMU and wheel-velocity sensors. The environment is a Gymnasium wrapper; it does not train a policy itself.

The environment defaults to a fast simulation step (`model.opt.timestep` from the MJCF). The PPO notebooks can pass `PPOConfig.timestep=0.0` for fast rollout or a nonzero timestep (the curriculum notebook uses `0.005` for its evaluation example) for real-time pacing. A rendering timestep affects wall-clock pacing, not the learned observation or action dimensions.

## 2. What the policy sees and controls

One observation is a `float32` vector in this exact order:

| Index | Value | Source/unit |
| --- | --- | --- |
| 0 | Estimated pitch | radians; complementary-filter estimate |
| 1 | Pitch rate | IMU gyroscope Y, radians/second |
| 2 | Left wheel velocity | MuJoCo wheel sensor, radians/second |
| 3 | Right wheel velocity | MuJoCo wheel sensor, radians/second |

The declared observation-space bounds are `[-π, -20, -50, -50]` to `[π, 20, 50, 50]`; the environment does not normalize observations. Pitch is estimated by combining the accelerometer angle `-atan2(accel_x, accel_z)` with the integrated gyroscope rate:

```text
pitch_t = alpha * (pitch_(t-1) + pitch_rate * dt)
          + (1 - alpha) * accelerometer_pitch
```

The default `alpha=0.99` weights the integrated gyro strongly while the accelerometer corrects long-term drift. Reset clears the filter state.

The action is a two-element `float32` vector in `[-1, 1]`: left and right motor control. The environment writes these values directly to the named MuJoCo actuator controls. It does not scale or clip arbitrary input inside `step`; the policy's Gaussian samples are not intrinsically bounded by the declared Box, so the trainer/environment integration and actuator limits matter. On hardware the firmware clamps and scales outputs before the motor driver; that mapping is separate from this training wrapper.

## 3. Episode reset and termination

`reset(seed=...)` resets MuJoCo, resets the pitch filter and step counter, and gives the robot a random initial angular velocity about its pitch axis (`qvel[4]` sampled uniformly from `[-0.5, 0.5]` rad/s). It calls `mj_forward` to refresh derived state and returns the first observation with an empty info dictionary. Gymnasium's seeded random generator makes this perturbation reproducible for a given seed.

An episode is terminated when the absolute estimated pitch exceeds the default 30-degree tip threshold. It is truncated when the step counter reaches `max_steps` (default 10,000). These are separate Gymnasium signals: tipping is task failure; hitting the configured horizon is a time limit.

## 4. Per-step reward and privileged simulator data

After applying the action and advancing MuJoCo one step, `step` computes:

```text
reward = alive_bonus
         - pitch_penalty_coef * pitch²
         - action_penalty_coef * sum(action²)
         - position_penalty_coef * (x² + y²)
         - yaw_penalty_coef * abs(yaw_rate)
```

The environment defaults are `alive_bonus=1.0`, pitch coefficient `5.0`, action coefficient `0.01`, position coefficient `0.01`, and yaw coefficient `0.1`. Notebooks override some coefficients for a training phase. There is no separate terminal penalty; termination occurs through the tip threshold and the agent receives the ordinary reward for that final step.

The observation is sensor-like, but position and yaw reward terms read `data.qpos[0:2]` and `data.qvel[5]` directly. These are privileged simulator state: they shape training but are not included as policy inputs and are not necessarily available from the physical robot. A policy can therefore be optimized using reward information that the deployed actor never observes. This is a sim-to-real limitation to account for when interpreting results.

## 5. Environment construction in the notebooks

The baseline and curriculum notebooks resolve the simplified MJCF path, construct `BalanceBotEnv`, wrap each instance with `RecordEpisodeStatistics`, and place four environments in a `gym.vector.SyncVectorEnv`. SyncVectorEnv steps its environments sequentially in one process; it batches the resulting observations for the policy but is not multiprocessing. In the baseline notebook only environment zero can render. Native macOS runs should disable rendering in the notebook kernel and use the documented `mjpython` route for viewer use.

The base PPO notebook sets seed 42, four environments, one million steps per environment (four million total), rollout length 2,048 per environment, 32 minibatches, 10 update epochs, initial learning rate `3e-4`, `gamma=0.99`, `gae_lambda=0.95`, policy clip `0.2`, value clip `1.0`, and max gradient norm `0.5`. It calls `train` once. Exact network sizes come from the config supplied by that notebook/trainer version; the curriculum notebook explicitly selects two hidden layers of 16 units for both actor and critic.

## 6. The PPO learning loop

The trainer is a customized continuous-action PPO implementation in `workspace/software/rl/ppo_trainer.py` (see the [trainer reference](ppo-trainer.md)). The actor is an MLP that emits a mean for each action dimension plus a learned log standard deviation; an independent Gaussian is sampled for each motor. The critic is an MLP that estimates the expected discounted return from an observation. Hidden layers use tanh activations and orthogonal initialization. The critic helps calculate training targets but is not needed for actor-only inference.

For each iteration, `train` collects `num_steps` observations from each environment, storing observations, actions, action log probabilities, rewards, episode-ending flags, and value estimates. With four environments and 2,048 steps, a rollout contains 8,192 transitions. At rollout end the critic estimates the value of the last observations. Generalized advantage estimation combines temporal-difference residuals using `gamma` and `gae_lambda`; returns are computed as advantages plus the rollout-time value estimates. At episode boundaries, the continuation trace is cut so one episode does not leak into another.

The flattened rollout is shuffled into minibatches and reused for up to `update_epochs`. PPO compares each stored action's new probability with its old probability and clips the probability ratio to `1 ± clip_coef` in the surrogate policy objective. The critic minimizes squared return error, with its value change limited by the separate `value_clip` when value clipping is enabled. The combined loss includes policy loss, `vf_coef * value_loss`, and an entropy term weighted by `ent_coef`; the curriculum/base settings use `ent_coef=0`. Gradients are clipped to `max_grad_norm` before Adam updates. When configured, approximate KL divergence can stop an update early. Learning rate annealing lowers the rate over training iterations.

This distinction is useful: simulation transitions supply experience; PPO does not backpropagate through MuJoCo physics. Gradients update the actor and critic using the stored batch and reward-derived targets.

## 7. Basic run versus curriculum run

### Single-stage baseline

`train_with_ppo.ipynb` trains from scratch under the environment's default reward coefficients. It uses the four-environment vector, runs `train(ppo_config, envs=envs)`, optionally reloads the best saved state into the in-memory agent, evaluates three episodes, exports TensorBoard scalars to CSV, then closes the environments. This notebook is one run; it does not itself implement the two-phase curriculum.

### Two-phase curriculum

`train_with_ppo_curriculum.ipynb` makes the sequence explicit:

1. **Learn to stay upright.** It constructs four environments with pitch coefficient `0.5`, action coefficient `0.01`, and position/yaw coefficients set to zero. With a `+1` alive reward, moving around can still be a viable balancing strategy because location and rotation are not penalized.
2. **Continue the same agent with extra objectives.** It changes the already-created environments' position coefficient to `0.001` and yaw coefficient to `0.1`, updates the experiment name, then calls `train(..., agent=result.agent)`. The actor and critic weights are carried forward; this is fine-tuning in a changed reward landscape, not a fresh policy. The trainer creates a new run/checkpoint directory for that call.

After each phase the notebook loads the best checkpoint into the returned agent when a best checkpoint exists, evaluates three episodes with `timestep=0.005`, exports TensorBoard data, and finally exports the final phase's best actor to ONNX. Curriculum effectiveness depends on completing cells in order and preserving the returned `result`; rerunning only a later cell may use stale variables or incompatible environment state.

Later notebooks extend this idea. The domain-randomization notebook labels phases 1–2 as balance then position/yaw shaping, phase 3 as observation noise and action delay, phase 4 as motor noise and random pushes, phase 5 as mass/friction variation, phase 6 as motor-gain variation, and phase 7 as random axle torque for tire-ridge disturbances. The command-conditioning notebook labels phases 1–2 the same way, then adds small velocity commands and pitch-rate shaping (phase 3), yaw and full velocity command ranges (phase 4), mid-episode command sampling (phase 5), then corresponding noise/delay, motor/push, mass/friction, motor-gain, and axle-torque stages (phases 6–10). Exact values live in those notebook cells and `balance_bot_env_dr.py` / `balance_bot_env_cmd.py`; do not infer a randomization schedule from the base curriculum.

## 8. What is saved and how to interpret it

Each `train` call creates a time-stamped directory below the chapter's `runs/` path, writes TensorBoard events, periodic state-dict checkpoints when enabled, a best model selected by recent episode return, and optionally a final state dict. `TrainResult` returns the agent, run path, checkpoint paths, and the best mean return observed by that run. The best file is based on episodic return and checkpoint interval; it is not a formal robustness or safety criterion. Curriculum phases have separate run directories, so a later phase's artifact is the one intended for final export.

The notebooks can export TensorBoard chart/loss scalars to CSV using a subprocess, avoiding a documented MuJoCo/TensorFlow LLVM conflict. Notebook output may reflect an earlier run, and output strings or return values are not a substitute for rerunning the current code. The base notebook's example evaluation and the curriculum notebook's recorded results differ; treat both as historical examples.

## 9. Evaluation and deployment pipeline

Evaluation uses the policy in inference mode, runs a configured number of episodes, and returns their episodic returns (with `NaN` for episodes reaching the evaluation limit as implemented). The curriculum notebook then exports only `actor_mean` as `actor.onnx`: the exported network maps a batch of four-value observations to two action means. The stochastic policy's learned Gaussian standard deviation and the critic are not exported. For embedded deployment the actor therefore acts as a deterministic mean-output network, followed by firmware clamping/scaling.

`workspace/software/rl/onnx_actor_to_c.py` converts the ONNX actor to a C header containing layer weights/biases and a forward routine. Chapter 03's firmware builds observations in the expected order and calls the generated actor. Keep the ONNX file and generated header paired. The firmware's sensor calibration, filtering, action scaling, motor direction, timing, and safety cutoff are additional control-system behavior, not learned by PPO.

## 10. Reproducibility and limitations

- Seed 42 is configured in the notebooks, but a seed does not guarantee bit-for-bit results across hardware, library versions, or execution backends.
- `SyncVectorEnv` provides multiple rollout streams but executes synchronously; `num_envs` multiplies samples per iteration and total configured transitions.
- Position/yaw reward shaping reads simulator-only state. It does not give that state to the actor.
- PPO Gaussian samples can fall outside the declared action Box; inspect the current wrapper and actuator limits when changing this integration.
- The best episodic-return checkpoint may not be the safest, smoothest, or most hardware-robust policy.
- Export and one-observation ONNX tests prove model-format execution only. They do not test closed-loop MuJoCo behavior, real sensor alignment, motor signs, or physical stability.
- Hardware calibration and controlled tests are required before a physical robot run. See the sim-to-real chapter documentation for its concrete checks.
