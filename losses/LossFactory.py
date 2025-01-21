from torch.nn import Module

import configs
import losses


class LossFactory:
    @staticmethod
    def create() -> Module:
        name = configs.LossConfig.NAME
        if name in losses.__dict__:
            model_class = getattr(losses, name)
        else:
            raise Exception(f"Loss {name} not found")

        try:
            model = model_class()
        except TypeError as e:
            raise TypeError(f"Could not instantiate {model_class}: {e}")

        return model
