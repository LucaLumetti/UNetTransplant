import os

import numpy as np
import torch
import torchio as tio

import configs
from datasets import DatasetFactory
from datasets.PatchDataset import PatchDataset
from experiments.BaseExperiment import BaseExperiment
from metrics.Metrics import Metrics
from models.modelFactory import ModelFactory
from taskvectors.TaskVector import TaskVector
from taskvectors.TiesMerging import TiesMerging


def main():
    # define linespace
    alpha_0 = np.linspace(0.80, 1.0, 10)  # [0.85]#
    alpha_1 = [1.0]  # np.linspace(0.30,0.50,10)#[0.36]#
    alpha_2 = np.linspace(0.2, 1.0, 20)
    best = 0.0

    for a0 in alpha_0:
        for a1 in alpha_1:
            for a2 in alpha_2:
                task_vector_lowerjaw = TaskVector(
                    checkpoints="/work/grana_maxillo/UNetMerging/checkpoints/TaskVectorTrainExperiment_ek90lzx4_taskvector_tf_mandible/epoch0020_2025-02-10 14:24:21.502936_task_vector.pth",
                )
                task_vector_lowerjaw.create_params_histogram()

                task_vector_pharynx = TaskVector(
                    checkpoints="/work/grana_maxillo/UNetMerging/checkpoints/TaskVectorTrainExperiment_ff3xd0ge_taskvector_tf_pharynx/epoch0020_2025-02-10 11:57:53.601929_task_vector.pth",
                )
                task_vector_pharynx.create_params_histogram()

                task_vector_canal = TaskVector(
                    checkpoints="/work/grana_maxillo/UNetMerging/checkpoints/TaskVectorTrainExperiment_37pgtnb8_taskvector_tf_lriac/epoch0020_2025-02-10 14:07:23.136610_task_vector.pth",
                )
                task_vector_combined = (
                    a0 * task_vector_lowerjaw
                    + a1 * task_vector_canal
                    + a2 * task_vector_pharynx
                )
                # task_vector_combined = task_vector_combined +

                # ties_vector_merged = TiesMerging(task_vector_combined)()

                # print(task_vector_lowerjaw.params[0])
                # print(task_vector_pharynx.params[0])
                # print(task_vector_combined.params[0])

                # dataset = DatasetFactory.create(split="val")
                dataset = PatchDataset(split="val", task=task_vector_combined.task)
                backbone, heads = task_vector_combined.get_backbone_and_heads(
                    tasks=[task_vector_combined.task]
                )
                backbone.eval()
                heads.eval()

                subject = dataset.dataset[1]
                # x = subject['images'][tio.DATA]
                # y = subject['labels'][tio.DATA]
                # i = subject['dataset_idx']

                backbone, heads = backbone.cuda(), heads.cuda()

                with torch.no_grad():
                    out = BaseExperiment.functional_predict(
                        backbone=backbone, heads=heads, subject=subject
                    )

                x = subject["images"][tio.DATA]
                y = subject["labels"][tio.DATA]
                metrics = Metrics(task=task_vector_combined.task).compute(out, y)
                print(f'Dice a0:{a0}, a1:{a1}, a2:{a2}: {metrics["dice"]}')

                if metrics["dice"].mean() > best:
                    best = metrics["dice"].mean()
                    print(
                        f'BEST Dice a0:{a0}, a1:{a1}, a2:{a2} --> Best Dice: {metrics["dice"]}'
                    )
                    os.makedirs("debug/merge", exist_ok=True)
                    np.save("debug/merge/image.npy", x[0])
                    np.save("debug/mergelabel.npy", y[0])
                    np.save(
                        "debug/mergepred.npy",
                        out.cpu().detach().numpy().astype(np.uint8),
                    )

                print("Done")


if __name__ == "__main__":
    configs.initialize_config("configs/merge_test.toml")
    main()
