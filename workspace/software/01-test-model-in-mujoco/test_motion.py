from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer


MJCF_PATH = Path(__file__).resolve().parents[2] / "mechanical/FreeCAD/bala2-fire/bala2-fire-simplified.xml"
MOTOR_SPEED_STEP = 0.1
MOTOR_SPEED_LIMIT = 1.0
PRINT_EVERY = 250

LEFT_MOTOR = "left_motor"
RIGHT_MOTOR = "right_motor"
IMU_ACCEL = "imu_accel"
IMU_GYRO = "imu_gyro"
IMU_ORIENTATION = "imu_orientation"
LEFT_WHEEL_POS = "left_wheel_pos"
RIGHT_WHEEL_POS = "right_wheel_pos"
LEFT_WHEEL_VEL = "left_wheel_vel"
RIGHT_WHEEL_VEL = "right_wheel_vel"

KEY_BACKSPACE = 259
KEY_UP = 265
KEY_DOWN = 264

ctrl = {"left": 0.0, "right": 0.0}


def clamp(value):
    return max(-MOTOR_SPEED_LIMIT, min(MOTOR_SPEED_LIMIT, value))


def format_sensor(values):
    return "[" + ", ".join(f"{value: .3f}" for value in values) + "]"


def render_status(lines):
    sys.stdout.write("\033[H")
    for line in lines:
        sys.stdout.write(f"\033[2K{line}\n")
    sys.stdout.write("\033[J")
    sys.stdout.flush()


def key_callback(keycode):
    if keycode == KEY_BACKSPACE:
        ctrl["left"] = ctrl["right"] = 0.0
    elif keycode == KEY_UP:
        ctrl["left"] = clamp(ctrl["left"] + MOTOR_SPEED_STEP)
        ctrl["right"] = clamp(ctrl["right"] + MOTOR_SPEED_STEP)
    elif keycode == KEY_DOWN:
        ctrl["left"] = clamp(ctrl["left"] - MOTOR_SPEED_STEP)
        ctrl["right"] = clamp(ctrl["right"] - MOTOR_SPEED_STEP)


def main():
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)
    left_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, LEFT_MOTOR)
    right_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, RIGHT_MOTOR)
    mujoco.mj_resetData(model, data)
    steps = 0
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0, 0, 0.05]
        viewer.cam.distance = 0.8
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -25

        while viewer.is_running():
            step_start = time.time()
            data.ctrl[left_motor_id] = ctrl["left"]
            data.ctrl[right_motor_id] = ctrl["right"]
            mujoco.mj_step(model, data)
            viewer.sync()

            steps += 1
            if steps % PRINT_EVERY == 0:
                render_status([
                    "MuJoCo: Manual motion test",
                    f"Model: {MJCF_PATH}",
                    f"Motor IDs: left {left_motor_id}, right {right_motor_id}",
                    "Controls: Up/Down change both motors; Backspace stops them; close viewer to quit.",
                    "Readout refresh: 2 Hz | sensor values rounded to 3 decimals",
                    "",
                    f"Accel:          {format_sensor(data.sensor(IMU_ACCEL).data)}",
                    f"Gyro:           {format_sensor(data.sensor(IMU_GYRO).data)}",
                    f"Orientation:    {format_sensor(data.sensor(IMU_ORIENTATION).data)}",
                    f"Wheel position: L {format_sensor(data.sensor(LEFT_WHEEL_POS).data)}  R {format_sensor(data.sensor(RIGHT_WHEEL_POS).data)}",
                    f"Wheel velocity: L {format_sensor(data.sensor(LEFT_WHEEL_VEL).data)}  R {format_sensor(data.sensor(RIGHT_WHEEL_VEL).data)}",
                    f"Motor command:  L {ctrl['left']:+.1f}  R {ctrl['right']:+.1f}",
                ])

            slack = model.opt.timestep - (time.time() - step_start)
            if slack > 0:
                time.sleep(slack)


if __name__ == "__main__":
    main()