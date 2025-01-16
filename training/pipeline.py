import torch
from tqdm import tqdm
from config import *

from training.dataloader.AllDatasetLoader import CombinedDataset
from training.losses.loss_factory import LossFactory
from training.models.model_factory import ModelFactory
from training.models.weight_init import init_weight_he
from training.optimizers.optim_factory import OptimizerFactory

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class Pipeline:
    def __init__(self,):
        # TODO: do not make them return a class, but the instance? mhhh look at optim, he need model params, maybe a partial is better
        optimizer_class = OptimizerFactory(optimizer_name=TRAIN_OPTIM_NAME).create()
        model_class = ModelFactory(model_name=TRAIN_MODEL_NAME).create()
        loss_class = LossFactory(loss_name=TRAIN_LOSS_NAME).create()
        scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR

        # data stuff
        combined_dataset = CombinedDataset()
        self.dataloader = torch.utils.data.DataLoader(
            combined_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=DATALOADER_NUM_WORKERS,
            pin_memory=True,
            # collate_fn=custom_collate_fn,
        ) 

        # model stuff
        self.model = model_class(in_channels=MODEL_IN_CHANNELS, out_channels=MODEL_OUT_CHANNELS).to("cuda")
        self.optimizer = optimizer_class(self.model.parameters())
        self.scheduler = scheduler_class(self.optimizer, T_max=TRAIN_EPOCHS)

        init_weight_he(self.model)

        self.loss = loss_class()

    def train(self):
        self.model.train()
        scaler = torch.GradScaler()

        for epoch in range(TRAIN_EPOCHS):
            losses = []
            for batch in tqdm(self.dataloader, total=len(self.dataloader), desc=f"Epoch {epoch+1}/{TRAIN_EPOCHS}"):
                self.optimizer.zero_grad()
                image = batch[0].to('cuda')
                label = batch[1].to('cuda')
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred = self.model(image)
                    loss = self.loss(pred, label)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                losses.append(loss.item())
                tqdm.write(f"Loss: {loss.item()}")

            self.scheduler.step()

            avg_loss = sum(losses) / len(losses)

            print(f"Epoch {epoch+1}/{TRAIN_EPOCHS} Loss: {avg_loss}")