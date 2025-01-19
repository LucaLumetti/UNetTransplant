import argparse

import wandb

from experiments import ExperimentFactory


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--experiment",
        help="Name of the experiment, supported: 'pretrain'.",
        default="PretrainExperiment",
    )
    args = arg_parser.parse_args()

    wandb.init(
        project="UNetMErging",
        name="test_nico",
        entity="maxillo",
    )

    experiment = ExperimentFactory.create(name=args.experiment)
    experiment.train()


if __name__ == "__main__":
    main()
