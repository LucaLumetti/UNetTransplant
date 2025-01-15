class Config:
    def __init__(self):
        self.batch_size = 2
        self.epochs = 100
        self.lr = 1e-3
        self.optimizer = "Adam"
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.num_workers = 4
        self.pin_memory = True
        self.save_dir = "models"
        self.log_dir = "logs"
        self.model_name = "unet3d"
        self.seed = 42
        self.in_channels = 1
        self.out_channels = 2

    def __str__(self):
        return (
            f"batch size: {self.batch_size}\n"
            f"epochs: {self.epochs}\n"
            f"lr: {self.lr}\n"
            f"optimizer: {self.optimizer}\n"
            f"momentum: {self.momentum}\n"
            f"weight_decay: {self.weight_decay}\n"
            f"num_workers: {self.num_workers}\n"
            f"pin_memory: {self.pin_memory}\n"
            f"save_dir: {self.save_dir}\n"
            f"log_dir: {self.log_dir}\n"
            f"model_name: {self.model_name}\n"
            f"seed: {self.seed}\n"
            f"in_channels: {self.in_channels}\n"
            f"out_channels: {self.out_channels}\n"
        )