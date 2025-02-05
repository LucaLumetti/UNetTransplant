import os
import torchio as tio
import numpy as np
import torch

import configs
from datasets import DatasetFactory
from datasets.PatchDataset import PatchDataset
from experiments.baseExperiment import BaseExperiment
from models.modelFactory import ModelFactory
from taskvectors.TaskVector import TaskVector
from taskvectors.TiesMerging import TiesMerging
from metrics.Metrics import Metrics


def main():
    #define linespace
    alpha_0 = np.linspace(0, 1, 5)
    alpha_1 = np.linspace(0, 1, 5)
    best=0.
    for a_0 in alpha_0:
        for a_1 in alpha_1:
            task_vector_lowerjaw = TaskVector(
                task_name="Lower Jaw",
                # checkpoints="/homes/gcapitani/UNetMerging/checkpoints/TaskVector_/work/grana_maxillo/UNetMerging/configs/taskvector_tf_mandible.toml/epoch0010_2025-02-04 22:36:44.240215_task_vector.pth",
                checkpoints="/homes/gcapitani/UNetMerging/checkpoints/TaskVector_/work/grana_maxillo/UNetMerging/configs/finetune_tf_mandible.toml/epoch0010_2025-02-04 16:09:20.201439_task_vector.pth",
                alphas=[a_0],
            )
            task_vector_lowerjaw.create_params_histogram()

            task_vector_pharynx = TaskVector(
                task_name="Pharynx",
                #checkpoints="/homes/gcapitani/UNetMerging/checkpoints/TaskVector_/work/grana_maxillo/UNetMerging/configs/taskvector_tf_pharynx.toml/epoch0010_2025-02-05 00:06:33.074498_task_vector.pth",
                checkpoints="/homes/gcapitani/UNetMerging/checkpoints/TaskVector_/work/grana_maxillo/UNetMerging/configs/finetune_tf_pharynx.toml/epoch0010_2025-02-04 22:25:07.191556_task_vector.pth",
                alphas=[a_1],
            )
            task_vector_pharynx.create_params_histogram()
            task_vector_combined = task_vector_lowerjaw + task_vector_pharynx

            # ties_vector_merged = TiesMerging(task_vector_combined)()

            #print(task_vector_lowerjaw.params[0])
            #print(task_vector_pharynx.params[0])
            #print(task_vector_combined.params[0])

            # dataset = DatasetFactory.create(split="val")
            dataset = PatchDataset(split="val", dataset_name="ToothFairy2")
            backbone, heads = task_vector_combined.get_backbone_and_heads(
                tasks=dataset.get_tasks()
            )
            backbone.eval()
            heads.eval()

            subject = dataset.dataset[0]
            # x = subject['images'][tio.DATA]
            # y = subject['labels'][tio.DATA]
            # i = subject['dataset_idx']

            backbone, heads = backbone.cuda(), heads.cuda()

            with torch.no_grad():
                out = BaseExperiment.functional_predict(
                    backbone=backbone, heads=heads, subject=subject
                )

            x = subject['images'][tio.DATA]
            y = subject['labels'][tio.DATA]
            metrics = Metrics().compute(out, y)
            print(f'Dice: {metrics["dice"]}')

            if metrics["dice"]>best:
                best=metrics["dice"]
                print(f'Alpha_0:{a_0} - Alpha_1_{a_1} --> Best Dice: {best}')
                np.save("debug/merge_image.npy", x[0])
                np.save("debug/merge_label.npy", y[0])
                np.save("debug/merge_pred.npy", out.cpu().detach().numpy().astype(np.uint8))

            print("Done")

if __name__ == "__main__":
    configs.initialize_config("/homes/gcapitani/UNetMerging/configs/merge.toml")
    main()
