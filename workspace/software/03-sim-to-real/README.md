# Sim-to-Real Balance Bot

This chapter connects the balance-bot policy trained in simulation to inference code that can be tested on a computer and then run on the robot. It includes a one-observation ONNX test, Arduino firmware, sensor checks, and IMU calibration.

## Files and Folders

| Path | Purpose |
| --- | --- |
| `inference_test.ipynb` | Notebook that loads a trained ONNX actor, runs one example observation, and displays the predicted action. |
| `inference_test.py` | Terminal version of the same one-shot ONNX inference test. Automatically uses the newest `actor.onnx` found under the PPO `runs` directory, or accepts an explicit model path. |
| `balance_bot/` | Main robot firmware. Reads the IMU and wheel encoders, prepares the policy observation, runs the generated C actor network, and drives the two motors. |
| `balance_bot/actor.h` | Generated C weights and `actor_forward` implementation corresponding to the ONNX actor. Regenerate it when deploying a different trained actor. |
| `balance_bot/bala.cpp`, `balance_bot/bala.h` | Shared Bala 2 motor, encoder, and I2C interface used by the Arduino sketches. |
| `balance_bot_fixes/` | Tuned firmware variant with additional pitch offset, motor boost, deadband, and output filtering settings. |
| `imu_calibration/` | Sketch for calibrating the IMU and saving offsets to non-volatile storage. |
| `sensor_test/` | Diagnostic sketch that reads the IMU and wheel encoders and prints pitch and wheel-velocity measurements over Serial. |
| `static_inference_test/` | Sketch that runs the C actor once on a fixed observation and prints the action and inference time. |

## The Inference Test

The Python inference test is a software-only check. It does not open MuJoCo, connect to the Bala 2, or send commands to motors. It feeds one four-value observation to the actor model and prints the two predicted action values.

The observation order matches the training/export code and firmware:

1. `pitch` in radians
2. `pitch_rate` in radians per second
3. `left_wheel_velocity` in radians per second
4. `right_wheel_velocity` in radians per second

The default test observation is `[0.2, 1.0, -0.5, 0.5]`. The model's output is the actor network's prediction for the left and right motors. These are raw network outputs, not a hardware command: the live firmware clamps each value to `[-1, 1]`, applies motor direction and scaling, and sends the resulting values to the motor driver.

## Prepare a Model

The Python test requires an exported `actor.onnx`. The script searches recursively under:

```text
workspace/software/02-balance-bot-with-ppo/runs/
```

It chooses the most recently modified `actor.onnx`. The curriculum PPO notebook exports the actor into its training checkpoint directory. Train and export an actor before running this test, or provide a model file explicitly with `--model`.

To convert an exported actor to the C header used by the firmware, run the repository converter from the root. Replace `YOUR_RUN` with the directory containing the exported model:

```sh
.venv/bin/python workspace/software/rl/onnx_actor_to_c.py \
  workspace/software/02-balance-bot-with-ppo/runs/YOUR_RUN/actor.onnx \
  workspace/software/03-sim-to-real/balance_bot/actor.h
```

If you deploy the `balance_bot_fixes` or `static_inference_test` sketch, generate or copy the matching header into that sketch's folder as well. Keep each header paired with the ONNX model from which it was generated.

## Run from a Terminal

Install the project dependencies in the repository's `.venv` as described in the main README. From the repository root, run:

```sh
.venv/bin/python workspace/software/03-sim-to-real/inference_test.py
```

To test a particular model and observation:

```sh
.venv/bin/python workspace/software/03-sim-to-real/inference_test.py \
  --model workspace/software/02-balance-bot-with-ppo/runs/YOUR_RUN/actor.onnx \
  --observation 0.2 1.0 -0.5 0.5
```

The output identifies the model and ONNX input/output, prints each observation component by name, and prints the left/right action values to four decimal places. If no model is found, the command reports the searched directory and explains that an export or `--model` path is needed.

## Run the Notebook

Open `inference_test.ipynb` in Jupyter and run its cells from top to bottom. Select a kernel that has `numpy` and `onnxruntime` installed. The notebook searches for the newest actor under the same PPO `runs` directory and uses the same default observation and labeled output format as the terminal script. Open Jupyter with this folder as its working directory so the notebook can locate the sibling PPO training directory.

## Arduino Firmware Workflow

The Python/ONNX check verifies that a model can execute on the development computer. To run inference on the robot, the matching actor weights must also be generated as C and included by the firmware. The main balance sketch constructs observations in the order listed above, calls `actor_forward`, clamps the two actions, scales them to motor-driver values, and sends them through the Bala interface.

Suggested hardware check sequence:

1. Upload `imu_calibration/imu_calibration.ino` and follow its Serial Monitor instructions. The robot must be held still during gyro calibration and placed on each indicated face during accelerometer calibration. Calibration is stored on the device.
2. Run `sensor_test/sensor_test.ino` to inspect pitch, pitch rate, and wheel velocities before enabling autonomous balancing.
3. Run `static_inference_test/static_inference_test.ino` to compare the C actor's action on its fixed test observation and measure its execution time.
4. Upload the desired live firmware from `balance_bot/` or `balance_bot_fixes/` only after the sensor readings and inference test are understood.

The firmware uses a 115200 baud Serial connection. Its debug prints are disabled by default because frequent serial output can disrupt the fast control loop. Follow the selected sketch's board/library configuration and verify motor direction, sensor signs, and safe startup conditions before testing with the robot on the ground. Keep the wheels clear of obstacles during initial tests and be ready to stop power if the bot behaves unexpectedly.

## What to Observe

- In the Python test, each named input is visible next to the actor's predicted left and right outputs. Changing `--observation` lets you check how different pitch, rotation-rate, and wheel-speed values affect the prediction.
- The static C test reports both inference duration and action values, allowing a direct check of the generated C network on the board.
- The sensor test reports the measurements that become policy inputs. Incorrect signs, units, calibration, or wheel encoder directions can make otherwise valid inference unsuitable for the physical robot.
- The live firmware repeatedly measures state, estimates pitch, runs the actor, and commands the motors. When the safety tip threshold is exceeded, it stops the wheels; the tuned variant adds filtering and response adjustments.

The one-shot Python inference test is not a simulator and does not prove that a policy is safe or stable on physical hardware. Validate the model, sensor conventions, motor directions, and safety behavior separately before deployment.