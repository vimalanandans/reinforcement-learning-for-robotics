# Shared PPO trainer reference

Source: `workspace/software/rl/ppo_trainer.py`. This module implements a customizable continuous-action PPO loop, originally based on CleanRL's continuous-action PPO, adapted for import from notebooks, custom Gymnasium environments, configurable MLP sizes, distinct value clipping, checkpointing, self-contained evaluation, and ONNX export.

## Public API

| Symbol | Purpose |
| --- | --- |
| `PPOConfig` | Dataclass for experiment, device, network, rollout, optimizer, loss, checkpoint, and pacing settings. Runtime-derived batch sizes/iteration counts are filled during training. |
| `Agent(envs, config)` | Builds actor and critic for the supplied vector environment's observation/action dimensions. |
| `TrainResult` | Returns in-memory agent, checkpoint directory, best/final model paths, and best recorded mean return. |
| `train(config, envs=None, agent=None)` | Collects rollouts and performs PPO optimization. A custom vector environment may be supplied directly; an existing agent allows fine-tuning. |
| `evaluate(agent, eval_episodes, config, envs=None, make_env=None)` | Runs policy episodes and returns returns; can use a provided vector env or construct one through an environment factory. |
| `build_mlp(...)` | Creates orthogonally initialized tanh hidden layers and an output layer. |
| `export_actor_onnx(...)` | Loads actor-mean weights from a saved state dict and exports the actor network. |
| `export_tb_plots_as_csv(run_path)` | Reads scalar TensorBoard events in a subprocess and writes grouped CSVs. |

`make_env(...)` creates a registered Gymnasium environment and applies wrappers for episode statistics, clipping, observation/reward normalization and clipping, and optional video capture. The balancing notebooks build their own `BalanceBotEnv` instances and wrap only episode statistics; they do not use this generic normalized-reward factory.

## Configuration groups

- **Run/device:** experiment name, seed, deterministic torch option, CUDA/MPS selection, optional tracking/video, checkpoint interval, final-save toggle, simulation timestep.
- **Network:** actor and critic hidden-layer counts and widths.
- **Rollout:** environment count, total timesteps, steps per environment, `gamma`, and GAE lambda.
- **Optimization:** learning rate and annealing, minibatches, update epochs, advantage normalization, policy clip, value clipping enable/bound, entropy/value coefficients, gradient norm limit, optional KL threshold.

The config values in a notebook override dataclass defaults. For accurate results, read the particular notebook's values alongside this API.

## Actor and critic

`Agent` infers flattened observation size and action dimension from the vector spaces. The critic maps observations to one scalar value. The actor mean network maps observations to action means; a separate trainable `actor_logstd` provides one learned log standard deviation per action dimension, broadcast across batch rows. Exponentiating log standard deviation keeps standard deviations positive. Sampling uses independent Normal distributions and sums per-dimension log probabilities and entropy for each complete action vector.

No tanh squashing is applied to sampled actions by `Agent`. A Box space declaration alone does not bound samples from an ordinary Normal distribution. Generic `make_env` wraps actions with `ClipAction`, while the balancing notebooks use their custom environment directly. Consider this integration detail when changing policy or actuator behavior.

## Training lifecycle

`train` seeds Python/NumPy/Torch, chooses an available device, computes batch dimensions, creates or accepts a vector environment and agent, and initializes Adam and TensorBoard logging. Each iteration optionally anneals learning rate, collects a rollout, computes bootstrapped GAE advantages/returns, shuffles flattened samples, and performs PPO minibatch optimization. It logs learning rate, policy/value losses, entropy, approximate KL values, clip fraction, explained variance, and throughput. An optional `target_kl` can stop further minibatch updates for an iteration.

Periodic checkpoints are named `checkpoint_iterNNNN.cleanrl_model`. At a checkpoint boundary, if recent episode returns are available, the trainer saves `best_model.cleanrl_model` when their mean exceeds the previous best. `save_model=True` writes `<exp_name>_final.cleanrl_model` at the end. State dicts contain actor and critic weights and actor log standard deviations. When `agent` is passed, its weights continue from that agent while the call initializes a new run and optimizer; optimizer state is not returned or passed through the agent parameter.

## Evaluation and exports

Evaluation chooses an action through the actor in evaluation mode and gathers episodic returns. The `timestep` config can adjust simulator pacing for real-time visualization. The actor ONNX exporter rebuilds the actor-mean MLP, extracts only `actor_mean.*` keys from the saved state dict, and exports a batch-capable graph named `observation` to `action`. It excludes critic and stochastic sampling. The CSV exporter launches a fresh Python process to avoid a TensorBoard/TensorFlow and MuJoCo LLVM interaction noted in the source.

## Operational cautions

- Environment wrappers materially affect what the agent observes and what rewards it learns; distinguish generic `make_env` from custom notebook factories.
- Loading a checkpoint into an agent restores model weights, but a subsequent `train` call creates a new optimizer rather than restoring optimizer moments.
- Best-model selection is based on recent training episode return, not a separate validation suite.
- ONNX actor means should be interpreted with the deployment's action clamp/scale and sensor preprocessing in view.
