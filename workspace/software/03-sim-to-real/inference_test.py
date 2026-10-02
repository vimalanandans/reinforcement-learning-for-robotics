import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort


SOFTWARE_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = SOFTWARE_DIR / "02-balance-bot-with-ppo" / "runs"
DEFAULT_OBSERVATION = [0.2, 1.0, -0.5, 0.5]
OBSERVATION_LABELS = ["pitch", "pitch_rate", "left_wheel_velocity", "right_wheel_velocity"]
ACTION_LABELS = ["left_motor", "right_motor"]


def find_latest_actor():
    candidates = list(RUNS_DIR.glob("**/actor.onnx"))
    if not candidates:
        raise FileNotFoundError(
            f"No actor.onnx model found under {RUNS_DIR}. "
            "Run the PPO curriculum training notebook to export a model, "
            "or pass a model path with --model."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Run one observation through a trained balance-bot ONNX actor.")
    parser.add_argument("--model", type=Path, help="Path to actor.onnx (default: newest export under the PPO runs folder).")
    parser.add_argument(
        "--observation",
        nargs=4,
        type=float,
        default=DEFAULT_OBSERVATION,
        metavar=("PITCH", "PITCH_RATE", "LEFT_WHEEL_VEL", "RIGHT_WHEEL_VEL"),
        help="Observation values in the model's expected order.",
    )
    args = parser.parse_args()

    try:
        model_path = (args.model or find_latest_actor()).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file does not exist: {model_path}")

        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        model_input = session.get_inputs()[0]
        model_output = session.get_outputs()[0]
        observation = np.asarray([args.observation], dtype=np.float32)
        action = session.run([model_output.name], {model_input.name: observation})[0].reshape(-1)
    except (FileNotFoundError, IndexError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    print("ONNX inference test")
    print(f"Model: {model_path}")
    print(f"Input: {model_input.name} {tuple(model_input.shape)}")
    print("Observation:")
    for label, value in zip(OBSERVATION_LABELS, args.observation):
        print(f"  {label:22s} {value:+.4f}")
    print(f"Output: {model_output.name}")
    print("Predicted action:")
    for index, value in enumerate(action):
        label = ACTION_LABELS[index] if index < len(ACTION_LABELS) else f"action_{index}"
        print(f"  {label:22s} {value:+.4f}")
    print("Note: This runs model inference only; it does not open MuJoCo or command hardware.")


if __name__ == "__main__":
    main()