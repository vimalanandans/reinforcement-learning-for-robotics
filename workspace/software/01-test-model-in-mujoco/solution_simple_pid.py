import math
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer


MJCF_PATH = Path(__file__).resolve().parents[2] / "mechanical/FreeCAD/bala2-fire/bala2-fire-simplified.xml"
MOTOR_SPEED_LIMIT = 1.0
PRINT_EVERY = 250

LEFT_MOTOR = "left_motor"
RIGHT_MOTOR = "right_motor"
IMU_ACCEL = "imu_accel"
IMU_GYRO = "imu_gyro"
IMU_ORIENTATION = "imu_orientation"

ALPHA = 0.99
KP = 7.0
KD = 0.5
TIP_THRESHOLD = math.radians(30)


def clamp(value):
    return max(-MOTOR_SPEED_LIMIT, min(MOTOR_SPEED_LIMIT, value))


def render_status(lines):
    sys.stdout.write("\033[H")
    for line in lines:
        sys.stdout.write(f"\033[2K{line}\n")
    sys.stdout.write("\033[J")
    sys.stdout.flush()


def main():
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)
    left_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, LEFT_MOTOR)
    right_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, RIGHT_MOTOR)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    steps = 0
    pitch = 0.0
    tipped = False
    prev_time = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0, 0, 0.05]
        viewer.cam.distance = 0.8
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -25

        while viewer.is_running():
            step_start = time.time()

            if data.time < prev_time:
                tipped = False
                mujoco.mj_forward(model, data)
                accel_x, _, accel_z = data.sensor(IMU_ACCEL).data
                pitch = -math.atan2(accel_x, accel_z)
            prev_time = data.time

            accel_x, _, accel_z = data.sensor(IMU_ACCEL).data
            pitch_rate = data.sensor(IMU_GYRO).data[1]
            accel_pitch = -math.atan2(accel_x, accel_z)
            pitch = ALPHA * (pitch + pitch_rate * model.opt.timestep) + (1 - ALPHA) * accel_pitch

            if abs(pitch) > TIP_THRESHOLD:
                tipped = True
            motor_speed_target = 0.0 if tipped else clamp((KP * pitch) + (KD * pitch_rate))
            data.ctrl[left_motor_id] = motor_speed_target
            data.ctrl[right_motor_id] = motor_speed_target

            mujoco.mj_step(model, data)
            viewer.sync()

            steps += 1
            if steps % PRINT_EVERY == 0:
                render_status([
                    "MuJoCo: PID balance controller",
                    f"Model: {MJCF_PATH}",
                    f"Motor IDs: left {left_motor_id}, right {right_motor_id}",
                    "Close the viewer to quit; Backspace resets the simulation.",
                    "Readout refresh: 2 Hz | angles and rates rounded to 3 decimals",
                    "",
                    f"Accel pitch:    {accel_pitch:+.3f} rad",
                    f"Filtered pitch: {pitch:+.3f} rad",
                    f"Pitch rate:     {pitch_rate:+.3f} rad/s",
                    f"Motor command:  {motor_speed_target:+.3f}",
                    f"Tipped:         {tipped}",
                ])

            slack = model.opt.timestep - (time.time() - step_start)
            if slack > 0:
                time.sleep(slack)

    print()


if __name__ == "__main__":
    main()