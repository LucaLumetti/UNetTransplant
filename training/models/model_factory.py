from config import MODEL_IN_CHANNELS, MODEL_OUT_CHANNELS


class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def create(self):
        if self.model_name == "unet3d":
            from training.models.unet3d.unet3d import UNet3D

            return UNet3D
        else:
            raise ValueError(f"Model {self.model_name} not supported.")
