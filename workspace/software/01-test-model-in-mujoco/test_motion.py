from pathlib import Path
import time

import mujoco
import mujoco.viewer


MJCF_PATH = Path(__file__).resolve().parents[2] / "mechanical/FreeCAD/bala2-fire/bala2-fire-simplified.xml"
MOTOR_SPEED_STEP = 0.1
MOTOR_SPEED_LIMIT = 1.0
PRINT_EVERY = 50

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
    print(f"Model: {MJCF_PATH}")
    print(f"Left motor ID: {left_motor_id}; right motor ID: {right_motor_id}")
    print("Use Up/Down arrows to change motor speed; Backspace resets the motors.")

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
                print(
                    f"Accel: {data.sensor(IMU_ACCEL).data} | "
                    f"Gyro: {data.sensor(IMU_GYRO).data} | "
                    f"Orientation: {data.sensor(IMU_ORIENTATION).data} | "
                    f"Wheels pos: {data.sensor(LEFT_WHEEL_POS).data}, "
                    f"{data.sensor(RIGHT_WHEEL_POS).data} | "
                    f"vel: {data.sensor(LEFT_WHEEL_VEL).data}, "
                    f"{data.sensor(RIGHT_WHEEL_VEL).data} | "
                    f"motors: {ctrl['left']:.1f}, {ctrl['right']:.1f}"
                )

            slack = model.opt.timestep - (time.time() - step_start)
            if slack > 0:
                time.sleep(slack)


if __name__ == "__main__":
    main()