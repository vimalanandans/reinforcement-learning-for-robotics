# Test the Robot Model in MuJoCo

This folder contains two demonstrations of the two-wheeled balance bot in MuJoCo. Both load the same MJCF model at `workspace/mechanical/FreeCAD/bala2-fire/bala2-fire-simplified.xml`, simulate its motors and sensors, and display the robot in the MuJoCo viewer.

The demos are intentionally different:

- `test_motion` is an open-loop, manually controlled test. You change the wheel motor commands with the keyboard and observe how the robot moves and falls.
- `solution_simple_pid` is a closed-loop balance controller. It estimates the robot's pitch from simulated IMU readings and commands both motors to keep the robot upright.

Each demo is available both as a notebook (`.ipynb`) and as a terminal script (`.py`). The notebook and script with the same base name implement the same demonstration. Use the notebooks for exploration in Jupyter, or run the Python scripts directly when you want to start the native MuJoCo viewer from a terminal.

## Files

| File | Purpose |
| --- | --- |
| `test_motion.ipynb` | Notebook for loading the model, inspecting actuators and sensors, and manually driving both wheels. |
| `test_motion.py` | Terminal version of the manual motion test. It prints IMU, wheel, and motor readings while the viewer runs. |
| `solution_simple_pid.ipynb` | Notebook for examining and running the complementary-filter and PD balance controller. |
| `solution_simple_pid.py` | Terminal version of the PID balance demonstration. It prints the estimated pitch, pitch rate, motor command, and tipped state. |

The model file is `workspace/mechanical/FreeCAD/bala2-fire/bala2-fire-simplified.xml`. Its mesh and other model assets are stored alongside it. The scripts locate this model relative to their own file location, so they can be launched from the repository root or another working directory.

## Requirements

You need Python with `mujoco` installed. The repository's native macOS setup installs the dependencies from `requirements-macos.txt` into `.venv` at the repository root. The Docker environment also includes the software required by the demonstrations.

On macOS, use MuJoCo's `mjpython` launcher for scripts that open the native viewer. It prepares the GUI event loop required by the viewer; ordinary `python` is suitable for loading the model or running code without the viewer, but may not launch the native GUI correctly.

## Run the Terminal Scripts on macOS

From the repository root, activate the project environment and run either script with `mjpython`:

```sh
source .venv/bin/activate
mjpython workspace/software/01-test-model-in-mujoco/test_motion.py
```

For the automatic balance demonstration, use a second command after closing the first viewer:

```sh
mjpython workspace/software/01-test-model-in-mujoco/solution_simple_pid.py
```

The commands above use `mjpython` from the activated `.venv`. You can also run them without activating the environment by using `.venv/bin/mjpython` in place of `mjpython`.

## Run in Docker

Start the repository's Docker desktop environment as described in the main repository README. In a terminal inside the container, run:

```sh
cd /workspace/software/01-test-model-in-mujoco
python test_motion.py
```

To try the controller instead, close the first viewer and run:

```sh
python solution_simple_pid.py
```

In Docker, the MuJoCo viewer is displayed through the container's WebTop desktop. The native macOS `mjpython` launcher is not needed inside the Linux container.

## Run the Notebooks

Open either notebook in Jupyter and run its cells from top to bottom. The notebook kernel must use the environment where MuJoCo is installed. If the project kernel has not yet been registered, run this from the repository root:

```sh
.venv/bin/python -m ipykernel install --user --name rl-robotics --display-name "Python (.venv rl-robotics)"
```

Then select **Python (.venv rl-robotics)** as the notebook kernel. The notebook path setup expects Jupyter to use this folder as the notebook working directory, as it does when launched with the notebook path from this directory.

On native macOS, the final viewer cell launches the matching `.py` script as a child process using `mjpython` from the selected kernel's environment. The cell remains busy while the MuJoCo window is open and finishes when you close the viewer. Select the project `.venv` kernel so the notebook can find its `mjpython` executable. On Linux and in Docker, the notebook runs the viewer loop directly in the cell.

## Demo 1: Manual Motion Test

Run `test_motion` to send the same motor command to both wheels. The motor command starts at zero and is limited to the range from -1.0 to 1.0.

| Input | Effect |
| --- | --- |
| Up arrow | Increase both motor commands by 0.1. |
| Down arrow | Decrease both motor commands by 0.1. |
| Backspace | Reset the motor commands to zero. |
| Close viewer | End the simulation. |

Observe how changing the motor command moves the wheels and chassis, and how the bot behaves when it leans or falls. The terminal or notebook output reports simulated accelerometer and gyroscope readings, orientation, wheel positions, wheel velocities, and current motor commands. This is a manual experiment, not a balance controller: the bot may fall unless you adjust the motor speed in time.

## Demo 2: PID Balance Controller

Run `solution_simple_pid` to let the controller drive both wheels automatically. Each simulation step reads the model's accelerometer and gyroscope. A complementary filter combines accelerometer-derived pitch with integrated gyroscope pitch rate. A proportional-derivative (PD) controller uses that estimate and the pitch rate to calculate a shared motor command for both wheels. The command is clamped to the range from -1.0 to 1.0.

The demo stops commanding the wheels when the estimated pitch exceeds 30 degrees, marking the bot as tipped. This illustrates the controller's safety condition; it does not guarantee that the simulated bot will remain upright for every initial state or parameter choice. The displayed terminal readings include accelerometer pitch, filtered pitch, pitch rate, motor command, and whether the controller has marked the bot as tipped.

Close the viewer to stop. In the notebook, MuJoCo's Backspace reset resets the simulation data; the notebook loop notices the reset and clears its tipped state. In the terminal script, close the viewer to stop the process.

## What to Look For

- The chassis and wheels are simulated from the robot's MJCF model; the motion is not a prerecorded animation.
- The model's IMU sensors provide measurements that can be compared with the robot's visible tilt and motion.
- In the manual test, motor commands are inputs that you adjust; the controller does not try to balance the robot.
- In the PID test, the estimated pitch and pitch rate affect the motor command. Watch the command change as the bot leans.
- If the bot crosses the tip threshold, the PID demo commands zero motor speed until the simulation is reset.
- The simulation is paced to approximately match the timestep specified in the MJCF model, so the viewer can show the robot moving in real time.