import argparse as ap

from reinforce.config import RobotConfig, EnvConfig, RewardConfig, GUIConfig, ModelConfig
from reinforce.gui import GUI

def parse_args() -> ap.Namespace:
    p = ap.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--train", action="store_true", help="Train policy, save, then run test.")
    g.add_argument(
        "--test", action="store_true", help="Run test only (loads model from --model-path)."
    )

    p.add_argument(
        "--no-sim", action="store_true", help="(train only) Hide simulation panel; show plots only."
    )
    p.add_argument(
        "--extended", action="store_true", help="(train --no-sim only) Show all 7 training metrics."
    )
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--train-episodes", type=int, default=None)
    p.add_argument("--test-episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:

    args = parse_args()

    robot_cfg = RobotConfig()
    env_cfg = EnvConfig()
    rew_cfg = RewardConfig()
    gui_cfg = GUIConfig()
    model_cfg = ModelConfig()

    gui_cfg.train_episodes = args.train_episodes or gui_cfg.train_episodes
    gui_cfg.test_episodes = args.test_episodes or gui_cfg.test_episodes
    gui_cfg.model_path = args.model_path or gui_cfg.model_path

    start_mode = "train" if args.train else "test"
    no_sim_train = bool(args.no_sim) if args.train else False
    extended = bool(args.extended) if (args.train and args.no_sim) else False

    app = GUI(
        gui_cfg=gui_cfg,
        robot_cfg=robot_cfg,
        env_cfg=env_cfg,
        reward_cfg=rew_cfg,
        model_cfg=model_cfg,
        seed=int(args.seed),
        start_mode=start_mode,
        no_sim_train=no_sim_train,
        extended=extended,
    )
    app.run()


if __name__ == "__main__":
    main()
