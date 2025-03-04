import argparse
import os
import sys
from pathlib import Path

import torch
import wandb

import configs
from experiments import ExperimentFactory


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--experiment",
        help="Name of the experiment, supported: 'PretrainExperiment', 'TaskVectorTrainExperiment'.",
    )
    arg_parser.add_argument(
        "--config",
        help="Path to the configuration file.",
    )
    arg_parser.add_argument(
        "--expname",
        help="Experiment name to append, config_basename is used if not provided.",
        default=None,
    )
    arg_parser.add_argument(
        "--override",
        nargs="+",
        help="Override configuration values.",
    )  # values can be overrided by passing them as DataConfig.DATASET_CLASSES = ['1,2'], DataConfig.BATCH_SIZE = 1
    args = arg_parser.parse_args()

    if os.uname().nodename == "ailb-login-02":
        torch.backends.cudnn.enabled = False
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    config_path = find_config_path(args.config)
    config_basename = config_path.stem
    configs.initialize_config(config_path)
    if args.override:
        override_config(configs, args.override)

    postfix = args.expname if args.expname else config_basename

    experiment_name = f"{args.experiment}_{wandb.util.generate_id()}_{postfix}"  # type: ignore

    wandb_mode = "disabled"

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

    # experiment.train()
    experiment.evaluate()


def find_config_path(config: str) -> Path:
    config_path = Path(config)
    if not config_path.exists():
        config_path = Path("configs") / config_path
    if not config_path.exists() and ".toml" != config_path.suffix:
        return find_config_path(f"{config}.toml")
    return config_path


def override_config(configs, overrides: list):
    for override in overrides:
        override = override.replace(" =", "=")
        override = override.replace("= ", "=")
        key, value = override.split("=", 1)

        config_class = getattr(configs, key.split(".")[0])
        config_key = key.split(".")[1]
        type_of_value = type(getattr(config_class, config_key))
        if type_of_value == list:
            value = eval(value)
        else:
            value = type_of_value(value)
        setattr(config_class, config_key, value)
        print(f"Overriding {config_class.__class__.__name__}.{key} with {value}")


if __name__ == "__main__":
    main()
