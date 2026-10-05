import argparse
import sys
from pathlib import Path

import gymnasium as gym

from balance_bot_env import BalanceBotEnv


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR.parent))

from rl.ppo_trainer import PPOConfig, train


def make_envs(mjcf_path, num_envs, render, **kwargs):
    def make_one(render_this_env):
        env = BalanceBotEnv(
            mjcf_path=mjcf_path,
            render_mode="human" if render and render_this_env else None,
            **kwargs,
        )
        return gym.wrappers.RecordEpisodeStatistics(env)

    factories = [
        lambda render_this_env=(index == 0): make_one(render_this_env)
        for index in range(num_envs)
    ]
    return gym.vector.SyncVectorEnv(factories)


def main():
    parser = argparse.ArgumentParser(
        description="Train the balance bot with a live MuJoCo viewer."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Total environment steps per curriculum phase (default: 50000).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of environments; only the first is rendered (default: 4).",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable the viewer and run without real-time pacing.",
    )
    args = parser.parse_args()

    if args.num_envs < 1 or args.timesteps < args.num_envs * 2048:
        parser.error("--num-envs must be positive and --timesteps must allow at least one rollout")

    mjcf_path = WORKSPACE_DIR / "mechanical/FreeCAD/bala2-fire/bala2-fire-simplified.xml"
    if not mjcf_path.is_file():
        raise FileNotFoundError(f"MuJoCo model not found: {mjcf_path}")

    config = PPOConfig(
        exp_name="balance-bot-ppo-live",
        env_id="BalanceBot-v0",
        seed=42,
        num_envs=args.num_envs,
        actor_hidden_layers=2,
        actor_hidden_size=16,
        critic_hidden_layers=2,
        critic_hidden_size=16,
        total_timesteps=args.timesteps,
        num_steps=2048,
        num_minibatches=32,
        update_epochs=10,
        anneal_lr=True,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        value_clip=1.0,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        checkpoint_interval=50,
        save_model=True,
        timestep=1 / 30 if not args.no_render else None,
    )

    envs = make_envs(
        mjcf_path,
        args.num_envs,
        render=not args.no_render,
        pitch_penalty_coef=0.5,
        action_penalty_coef=0.01,
        position_penalty_coef=0.0,
        yaw_penalty_coef=0.0,
    )

    try:
        print(f"Model: {mjcf_path}")
        print(f"Viewer: {'off' if args.no_render else 'on (30 FPS)'}")
        print(f"Training for {args.timesteps:,} steps per curriculum phase")
        result = train(config, envs=envs)

        for env_wrapper in envs.envs:
            env = env_wrapper.env
            env.position_penalty_coef = 0.001
            env.yaw_penalty_coef = 0.1

        print("Starting curriculum phase 2: penalizing drift and yaw")
        result = train(config, envs=envs, agent=result.agent)
        print(f"Best model: {result.best_model_path}")
        print(f"Final model: {result.final_model_path}")
        print(f"Best mean return: {result.best_mean_return:.2f}")
    finally:
        envs.close()


if __name__ == "__main__":
    main()
