# Balance Bot with PPO

This folder contains the Gymnasium environment tests and PPO training notebooks for the two-wheel balance bot.

## Run with JupyterLab on macOS

From the repository root, activate the project environment and start JupyterLab:

```sh
source .venv/bin/activate
jupyter lab
```

Open `workspace/software/02-balance-bot-with-ppo/env_test.ipynb` and run its cells from top to bottom. The notebook resolves the MJCF model from the repository workspace instead of using the Docker-only `/workspace/...` path.

The Gymnasium environment check runs without rendering in the notebook kernel. In the random-policy test cell, macOS starts `env_test_viewer.py` using the `mjpython` executable next to the active kernel. This opens the native MuJoCo viewer for the 200-step episode. Select the project `.venv` as the notebook kernel so it can find the matching `mjpython`.

`env_test.py` is a terminal-only model inspection script. Run it from this directory to print the model's joint position and velocity addresses:

```sh
mjpython env_test.py
```

## Notebooks

- `env_test.ipynb`: environment checker, reset/step tests, and a random-policy viewer run.
- `env_test_vec.ipynb`: vectorized environment tests.
- `train_with_ppo.ipynb`: PPO training.
- `train_with_ppo_curriculum.ipynb`: curriculum-based PPO training.

For native macOS PPO training, disable MuJoCo viewer rendering when starting JupyterLab, as described in the repository root README. The native MuJoCo viewer must run under `mjpython`; ordinary Jupyter kernels cannot directly call `launch_passive` on macOS.

## Watch PPO Training on macOS

### Purpose of the added runner

The training notebook runs in a regular Jupyter Python kernel. On macOS, that kernel cannot create MuJoCo's native viewer: the viewer must be launched by `mjpython`. The added `train_with_ppo_curriculum_live.py` runner starts PPO and the viewer in the same `mjpython` process, so the window shows the first environment as the current policy collects each rollout. It follows the notebook's two curriculum phases; the notebook itself remains unchanged and is still the option for ordinary headless training and analysis.

The runner defaults to a short 50,000-step preview per phase, not the notebook's full 2,000,000-step run. It paces rendered simulation at 30 FPS so the viewer is watchable. This pacing is only for visualization and makes training slower. After confirming the viewer works, run the full job headlessly to remove the viewer and pacing overhead.

### Watch a short run

From a terminal, run:

```sh
cd workspace/software/02-balance-bot-with-ppo
../../../.venv/bin/mjpython train_with_ppo_curriculum_live.py
```

For the full run, use the same curriculum runner without rendering. This matches the notebook's 500,000 steps per environment with four environments:

```sh
python train_with_ppo_curriculum_live.py --no-render --timesteps 2000000
```

`--timesteps` is the total number of environment steps per curriculum phase. You can adjust the preview length, for example `--timesteps 16384` for two PPO rollouts per phase. Start TensorBoard separately with `tensorboard --logdir runs --port 6006` to monitor either run. The runner uses a distinct experiment name, so its TensorBoard logs are separate from notebook runs.
