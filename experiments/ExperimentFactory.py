import experiments
from experiments.BaseExperiment import BaseExperiment


class ExperimentFactory:
    @staticmethod
    def create(experiment: str, name) -> BaseExperiment:
        if experiment in experiments.__dict__:
            experiment_class = getattr(experiments, experiment)
        else:
            raise Exception(f"Experiment {experiment} not found")

        try:
            model = experiment_class(experiment_name=name)
        except TypeError as e:
            raise TypeError(f"Could not instantiate {experiment_class}: {e}")

        return model
