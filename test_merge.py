import os

import numpy as np
import torch

import configs
from datasets import DatasetFactory
from models.modelFactory import ModelFactory


def main():
    pretrain_backbone = ModelFactory.create(configs.BackboneConfig)
    backbone_liver, head_liver = ModelFactory.create_from_checkpoint(
        configs.LiverHeadConfig.PRETRAIN_CHECKPOINTS
    )
    backbone_kidney, head_kidney = ModelFactory.create_from_checkpoint(
        configs.KidneyHeadConfig.PRETRAIN_CHECKPOINTS
    )

    for model in [
        pretrain_backbone,
        backbone_liver,
        head_liver,
        backbone_kidney,
        head_kidney,
    ]:
        model.cuda()
        model.eval()

    # pretrained_backbone_params = pretrain_backbone.state_dict()
    # liver_backbone_params = backbone_liver.state_dict()
    # kidney_backbone_params = backbone_kidney.state_dict()

    # liver_head_params = head_liver.state_dict()
    # kidney_head_params = head_kidney.state_dict()

    train_dataset = DatasetFactory.create()
    train_loader = train_dataset.get_dataloader()
    task = train_dataset.get_tasks()
    with torch.no_grad():
        for i, sample in enumerate(train_loader):
            image = sample[0].cuda()
            label = sample[1].cuda()

            if 1 not in torch.unique(label) or 2 not in torch.unique(label):
                continue

            pretrained_backbone_out = pretrain_backbone(image)
            liver_backbone_out = backbone_liver(image)
            kidney_backbone_out = backbone_kidney(image)

            backbones = {
                "pretrain_backbone": pretrained_backbone_out,
                "liver_backbone": liver_backbone_out,
                "kidney_backbone": kidney_backbone_out,
            }

            heads = {"liver_head": head_liver, "kidney_head": head_kidney}

            maks = {
                "liver_head": [1, 2, 3],
                "kidney_head": [0, 2, 3],
            }

            debug_path = "debug/merge_test/"
            os.makedirs(debug_path, exist_ok=True)

            np.save(f"{debug_path}image.npy", image.cpu().detach().numpy())
            np.save(f"{debug_path}label.npy", label.cpu().detach().numpy())

            for backbone_name, backbone_out in backbones.items():
                for head_name, head in heads.items():
                    print(f"backbone: {backbone_name}, head: {head_name}")
                    head_out = head(backbone_out)
                    # head_out = torch.sigmoid(head_out)

                    mask = maks[head_name]
                    head_out[:, mask, :, :, :] = 0

                    head_out = torch.concatenate(
                        [
                            torch.zeros(
                                (
                                    head_out.shape[0],
                                    1,
                                    head_out.shape[2],
                                    head_out.shape[3],
                                    head_out.shape[4],
                                ),
                                device=head_out.device,
                            ),
                            head_out,
                        ],
                        axis=1,
                    )
                    head_out = head_out.argmax(dim=1)

                    name = f"{backbone_name}__{head_name}"
                    np.save(f"{debug_path}{name}.npy", head_out.cpu().detach().numpy())

            continue

        # liver_head_out = head_liver(liver_backbone_out)
        # kidney_head_out = head_kidney(kidney_backbone_out)


if __name__ == "__main__":
    main()
