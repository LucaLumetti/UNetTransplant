from datetime import datetime

import torch
import wandb
from tqdm import tqdm

from config import TrainConfig
from datasets import DatasetFactory
from experiments import BaseExperiment
from losses import LossFactory
from models import ModelFactory
from optimizers import OptimizerFactory

# TODO: support DDP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PretrainExperiment(BaseExperiment):
    def __init__(
        self,
    ):
        super(BaseExperiment, self).__init__()

        self.train_dataset = DatasetFactory.create()
        # TODO: add validation

        # TODO: is there the correct place?
        self.train_loader = self.train_dataset.get_dataloader()

        self.model = ModelFactory.create()
        self.model = self.model.cuda()

        wandb.watch(self.model, log="all", log_freq=100)

        self.optimizer = OptimizerFactory.create(self.model)
        self.loss = LossFactory.create()
        # self.metrics = Metrics(self.config.model['n_classes'])

    def save(self, epoch):
        now = datetime.now()
        checkpoint_filename = f"checkpoint_{epoch}_{now}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                # "scheduler_state_dict": self.scheduler.state_dict(),
            },
            checkpoint_filename,
        )
        print(f"Checkpoint saved at {checkpoint_filename}")

    def train(self):
        self.model.train()
        for epoch in range(TrainConfig.EPOCHS):
            for i, sample in tqdm(
                enumerate(self.train_loader),
                desc=f"Epoch {epoch}",
                total=len(self.train_loader),
            ):
                image = sample[0].to(device)
                label = sample[1].to(device)

                output = self.model(image)

                loss: torch.Tensor = self.loss(output, label)

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                tqdm.write(f"Loss: {loss.item()}")
                wandb.log(
                    {
                        "Train/Loss": loss.item(),
                        "Train/lr": self.optimizer.param_groups[0]["lr"],
                    }
                )

            # torch.cuda.empty_cache()

            # if epoch % 10 == 0:
            #     self.test(phase='Val')
            #     self.model.train()
            #     torch.save(self.model.state_dict(), os.path.join(self.results_path, f'epoch_{epoch}.pth'))

            # torch.cuda.empty_cache()

    # @torch.inference_mode()
    # def test(self, phase='Test'):
    # assert phase in ['Test', 'Val'], f'phase should be Test or Val, passed: {phase}'

    # num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    # dataset = self.test_dataset if phase == 'Test' else self.val_dataset

    # avg_metrics = defaultdict(list)
    # avg_loss = []
    # self.model.eval()
    # for idx, sample in tqdm(enumerate(dataset), desc=phase, total=len(dataset)):
    #     grid_sampler = tio.inference.GridSampler(
    #         sample,
    #         self.config.dataset['patch_size'],
    #         self.config.dataset['grid_overlap'],
    #     )

    #     pred_aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="hann")
    #     label_aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="hann")

    #     loader = DataLoader(
    #         grid_sampler,
    #         num_workers=num_workers,
    #         batch_size=1,
    #         pin_memory=True,
    #     )

    #     for j, patch in enumerate(loader):
    #         image = patch['image'][tio.DATA].float().to(device)
    #         label = patch['label'][tio.DATA].float().to(device)
    #         _, prediction = self.model(image)
    #         pred_aggregator.add_batch(prediction, patch[tio.LOCATION])
    #         label_aggregator.add_batch(label, patch[tio.LOCATION])

    #     prediction = pred_aggregator.get_output_tensor().unsqueeze(0)
    #     label = label_aggregator.get_output_tensor().unsqueeze(0)

    #     loss = self.loss(prediction, label).item()

    #     prediction = prediction.argmax(dim=1, keepdim=True).cpu()
    #     label = label.int().cpu()

    #     metrics = self.metrics(prediction, label)

    #     avg_loss.append(loss)
    #     for k,v in metrics.items():
    #         avg_metrics[k].append(v.cpu().numpy().item())

    # avg_loss = sum(avg_loss)/len(avg_loss)

    # for k, v in avg_metrics.items():
    #     avg_metrics[k] = sum(v)/len(v)

    # results ={
    #     f'{phase}/Loss': avg_loss,
    #     **{f'{phase}/{k}': v for k, v in avg_metrics.items()}
    # }

    # if self.debug:
    #     print(results)

    # wandb.log(results)

    # return avg_metrics
