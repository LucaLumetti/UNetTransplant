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
    arg_parser.add_argument(
        "--name",
        help="Name for the experiment",
        default=None,
    )
    args = arg_parser.parse_args()

    config_path = find_config_path(args.config)
    configs.initialize_config(config_path)

    if args.name is None:
        args.name = f"{args.experiment}_{wandb.util.generate_id()}"

    wandb_mode = "online"

    if "debugpy" in sys.modules:
        print("Setting batch size to 1 for debugging.")
        print("Disabling wandb.")
        configs.DataConfig.BATCH_SIZE = 1
        wandb_mode = "disabled"

    wandb.init(
        project="UNetMerging",
        name=args.name,
        entity="maxillo",
        mode=wandb_mode,
        config=configs.generate_config_json(),
    )

    experiment = ExperimentFactory.create(name=args.experiment)

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
