import argparse
import sys
from pathlib import Path

import wandb

import configs
from experiments import ExperimentFactory


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--experiment",
        help="Name of the experiment, supported: 'PretrainExperiment'.",
        # default="PretrainExperiment",
        # default="TaskVectorExperiment",
    )
    arg_parser.add_argument(
        "--config",
        help="Path to the configuration file.",
        # default="configs/taskvector_tf_mandible.toml",
        # default="/work/grana_maxillo/UNetMerging/configs/pretrain.toml",
    )
    args = arg_parser.parse_args()

    config_path = find_config_path(args.config)
    config_basename = config_path.stem
    configs.initialize_config(config_path)

    experiment_name = f"{args.experiment}_{wandb.util.generate_id()}_{config_basename}"

    wandb_mode = "online"

    if "debugpy" in sys.modules:
        configs.DataConfig.BATCH_SIZE = 1
        configs.DataConfig.NUM_WORKERS = 1
        wandb_mode = "disabled"
        print(f"[DEBUG] BATCH_SIZE: {configs.DataConfig.BATCH_SIZE}")
        print(f"[DEBUG] NUM_WORKERS: {configs.DataConfig.NUM_WORKERS}")
        print(f"[DEBUG] wandb mode: {wandb_mode}")

    wandb.init(
        project="UNetMerging",
        name=experiment_name,
        entity="maxillo",
        mode=wandb_mode,
        config=configs.generate_config_json(),
    )

    experiment = ExperimentFactory.create(
        experiment=args.experiment, name=experiment_name
    )

    experiment.train()
    experiment.evaluate()


def find_config_path(config: str) -> Path:
    config_path = Path(config)
    if not config_path.exists():
        config_path = Path("configs") / config_path
    if not config_path.exists():
        return find_config_path(f"{config}.toml")
    return config_path


if __name__ == "__main__":
    main()
