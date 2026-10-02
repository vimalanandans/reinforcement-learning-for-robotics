# Reinforcement Learning for Robotics

This repository holds the development environment and demos used in the Reinforcement Learning for Robotics video series, which can be found [here](https://www.youtube.com/watch?v=zsdceSTRBl4&list=PLYExBrZNJeQg&index=1).

> If you are looking for the workshop/webinar version of this demo, please use this repository: [github.com/ShawnHymel/workshop-reinforcement-learning-for-robotics](https://github.com/ShawnHymel/workshop-reinforcement-learning-for-robotics/)

<a href="https://www.youtube.com/watch?v=zsdceSTRBl4&list=PLYExBrZNJeQg&index=1">
  <img src=".images/rl-for-robotics-thumbnail-play.png" alt="Reinforcement Learning for Robotics" height="500">
</a>

## Installation

Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/). Make sure it is running before continuing to the next step.

Open a terminal and build the Docker image:

```sh
docker build -t rl-robotics -f Dockerfile.cpu .
```

Run the image:

```sh
docker run -it --rm -p 3000:3000 -p 6006:6006 -v "${PWD}/workspace:/workspace" --shm-size=2g rl-robotics
```

Notes:
 * Port 3000 is for the WebTop interface
 * Port 6006 is for TensorBoard
 * VS Code is memory hungry, so we bump the shared memory up to 2 GB

Browse to [http://localhost:3000/](http://localhost:3000/) to interact with WebTop. Open VS Code from the desktop and use it to open and run the notebooks. The container starts WebTop and TensorBoard; it does not start a JupyterLab server.

### Apple Silicon (M1/M2/M3)

Use the separate ARM64 Dockerfile for Apple Silicon Macs:

```sh
docker build --platform linux/arm64 -t rl-robotics-macos -f Dockerfile.macos .
docker run -it --rm -p 3000:3000 -p 6006:6006 -v "${PWD}/workspace:/workspace" --shm-size=2g rl-robotics-macos
```

Open [http://localhost:3000/](http://localhost:3000/) and run the MuJoCo notebook in the container's desktop. The viewer runs inside the Linux container and is displayed through WebTop. Docker Desktop on macOS does not expose the Apple GPU to this Linux container, so the container uses CPU computation.

To use the M3 GPU for PPO network computation, run the training code natively on macOS with an Apple-Silicon PyTorch build that supports MPS. The trainer automatically selects MPS when `PPOConfig.cuda` is enabled and MPS is available. MuJoCo simulation still runs on the CPU, and device transfers during rollout may limit or eliminate the speedup for this small policy network. The container remains useful for the WebTop desktop and viewer, but its Linux Python process cannot use MPS.

### Native macOS MPS Training

Install Python 3.12 for Apple Silicon, then create and activate a virtual environment from the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-macos.txt
```

Confirm PyTorch can use the M3 GPU:

```sh
python -c 'import torch; print("MPS available:", torch.backends.mps.is_available()); assert torch.backends.mps.is_available()'
```

Launch the PPO notebook locally with MuJoCo viewer rendering disabled. MuJoCo simulation stays on the CPU; the PPO network uses MPS when available.

```sh
cd workspace/software/02-balance-bot-with-ppo
MUJOCO_RENDER=0 jupyter lab train_with_ppo.ipynb
```

The training notebook resolves the model and source paths from the repository workspace. Set `PPOConfig.cuda=False` to force CPU computation. MuJoCo's native macOS viewer requires `mjpython`; disable notebook rendering as shown above when running Jupyter with the ordinary Python kernel.

To monitor training logs, open a second terminal, activate the virtual environment, and run this from the PPO notebook directory:

```sh
source .venv/bin/activate
tensorboard --logdir runs --port 6006
```

Then open [http://localhost:6006/](http://localhost:6006/).

## License

All software in this repository, unless otherwise noted, is licensed under the [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) license.
