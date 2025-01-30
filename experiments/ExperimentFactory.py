import experiments
from experiments.baseExperiment import BaseExperiment


class ExperimentFactory:
    @staticmethod
    def create(name: str) -> BaseExperiment:
        if name in experiments.__dict__:
            experiment_class = getattr(experiments, name)
        else:
            raise Exception(f"Experiment {name} not found")

        try:
            model = experiment_class()
        except TypeError as e:
            raise TypeError(f"Could not instantiate {experiment_class}: {e}")

        return model
