import argparse

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
        default="configs/default_config.toml",
    )
    args = arg_parser.parse_args()
    configs.initialize_config(args.config)
    wandb.init(
        project="UNetMerging",
        name=f"{args.experiment}_{wandb.util.generate_id()}",
        entity="maxillo",
        mode="disabled",
        config=configs.generate_config_json(),
    )

    print(configs.DataConfig.NAME)

    experiment = ExperimentFactory.create(name=args.experiment)
    if configs.BackboneConfig.PRETRAIN_CHECKPOINTS is not None:
        experiment.load(
            checkpoint_path=configs.BackboneConfig.PRETRAIN_CHECKPOINTS,
        )

    experiment.train()
    experiment.evaluate()


if __name__ == "__main__":
    main()
