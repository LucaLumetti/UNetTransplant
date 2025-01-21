import argparse
from pathlib import Path

import wandb

import configs
from experiments import ExperimentFactory


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--experiment",
        help="Name of the experiment, supported: 'PretrainExperiment'.",
        default="FinetuneExperiment",
    )
    arg_parser.add_argument(
        "--config",
        help="Path to the configuration file.",
        default="configs/finetune_tf_pharynx.toml",
    )
    args = arg_parser.parse_args()

    config_path = find_config_path(args.config)
    configs.initialize_config(config_path)

    wandb.init(
        project="UNetMerging",
        name=f"{args.experiment}_{wandb.util.generate_id()}",
        entity="maxillo",
        mode="online",
        config=configs.generate_config_json(),
    )

    experiment = ExperimentFactory.create(name=args.experiment)
    if configs.BackboneConfig.PRETRAIN_CHECKPOINTS is not None:
        experiment.load(
            checkpoint_path=configs.BackboneConfig.PRETRAIN_CHECKPOINTS,
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
