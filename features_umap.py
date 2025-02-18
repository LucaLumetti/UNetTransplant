import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchio as tio
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm

import configs
from datasets import DatasetFactory
from datasets.PatchDataset import PatchDataset
from experiments.BaseExperiment import BaseExperiment
from models.modelFactory import ModelFactory
from preprocessing.preprocessor import Preprocessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_umap(x):
    reducer = umap.UMAP(n_neighbors=15, low_memory=True, min_dist=0.05)
    embedding = reducer.fit_transform(x)

    return embedding


def kmeans(x, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x)
    return kmeans.labels_


def main():
    backbone = ModelFactory.create(configs.BackboneConfig)

    backbone_state_dict = torch.load(configs.BackboneConfig.PRETRAIN_CHECKPOINTS)[
        "backbone_state_dict"
    ]
    str_to_remove = "_orig_mod."
    backbone_state_dict = {
        k.replace(str_to_remove, ""): v for k, v in backbone_state_dict.items()
    }

    # remove the last 1x1 conv layer
    keys = list(backbone_state_dict.keys())
    for key in keys:
        if "final_conv" in key:
            del backbone_state_dict[key]

    backbone.load_state_dict(backbone_state_dict)

    dataset = PatchDataset(split="val", dataset_name="ToothFairy2")
    for subject in dataset.dataset:
        patch_sampler = tio.inference.GridSampler(
            subject,
            patch_size=(96, 96, 96),
            patch_overlap=0,
        )

        spatial_shape = subject["images"][tio.DATA][0].shape

        pred_aggregator = torch.zeros((64, *spatial_shape), device="cuda")

        for patch in tqdm(patch_sampler):
            # unsqueeze to add batch dimension
            image_patch = patch["images"][tio.DATA].unsqueeze(0).to("cuda")
            dataset_idx = patch["dataset_idx"].unsqueeze(0)
            location = patch[tio.LOCATION].unsqueeze(0)

            with torch.inference_mode():
                backbone_pred = backbone(image_patch).squeeze(0)

            pred_aggregator[
                :,
                location[0, 0] : location[0, 3],
                location[0, 1] : location[0, 4],
                location[0, 2] : location[0, 5],
            ] = backbone_pred

        bb_pred = pred_aggregator.cpu().numpy()
        np.save(
            f"{configs.DataConfig.OUTPUT_DIR}debug/umap/image_array.npy",
            subject["images"][tio.DATA].cpu().numpy(),
        )
        np.save(f"{configs.DataConfig.OUTPUT_DIR}debug/umap/pred.npy", bb_pred)

        kmeans_labels = kmeans(bb_pred.reshape(64, -1).T, 10)
        kmeans_labels = kmeans_labels.reshape(*bb_pred.shape[-3:])
        np.save(
            f"{configs.DataConfig.OUTPUT_DIR}debug/umap/kmeans_labels.npy",
            kmeans_labels + 1,
        )
        print("Kmeans computed 🎉")
        continue

        original_shape = bb_pred.shape
        features = bb_pred.cpu().numpy().reshape(64, -1).T
        umap_embedding = compute_umap(features)

        x_vals = umap_embedding[:, 0]
        y_vals = umap_embedding[:, 1]

        # norm_x = (x_vals - x_vals.min()) / (x_vals.max() - x_vals.min())
        # norm_y = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min())

        # print(norm_x.max(), norm_x.min())
        # print(norm_y.max(), norm_y.min())

        # hues = norm_x
        # values = norm_y
        # sats = np.ones_like(hues)

        # colors = [mcolors.hsv_to_rgb([h, s, v]) for h, s, v in zip(hues, sats, values)]

        # color_volume = np.array(colors).reshape(3, original_shape[-3], original_shape[-2], original_shape[-1])

        # np.save(f"debug/umap/{dataset_idx}_color_volume.npy", color_volume)

        # create and save figure
        # plt.scatter(x_vals, y_vals, s=10, c=colors, edgecolors='black')
        #
        # colors based on kmeans_labels using turbo colormap
        colors = y.flatten()
        colors = colors / colors.max()
        colors = plt.cm.turbo(colors)

        plt.scatter(x_vals, y_vals, s=10, c=colors, edgecolors="black")
        plt.savefig(f"debug/umap/{dataset_idx}_umap.png")


if __name__ == "__main__":
    configs.initialize_config("/work/grana_maxillo/UNetMerging/configs/umap.toml")
    main()
    print("done")
