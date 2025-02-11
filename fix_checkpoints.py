import os

import torch

from task.Task import Task


def fix_task_class(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path)
    if "task" in checkpoint.keys():
        task = checkpoint["task"][0]
        if "task.Task" in str(task.__class__):
            task_dict = task.__dict__
            if "label_ranges" not in task_dict:
                new_label_ranges = task.dict_to_label_ranges(task.labels_to_predict)
                print(
                    f"Missing label_ranges in {checkpoint_path}, adding {new_label_ranges}"
                )
                task_dict["label_ranges"] = new_label_ranges
            if "task_name" not in task_dict:
                print(f"Missing task_name in {checkpoint_path}, please provide it")
                task_dict["task_name"] = input("Enter task name: ")
            task = Task(
                dataset_name=task_dict["dataset_name"],
                label_ranges=task_dict["label_ranges"],
                task_name=task_dict["task_name"],
            )
            checkpoint["task"] = [task]
            torch.save(checkpoint, checkpoint_path)
            print(f"Fixed task class in {checkpoint_path}")
        else:
            print(f"Task class in {checkpoint_path} is already fixed")
    else:
        print(f"No task class in {checkpoint_path}")


# All checkpoints have the key "heads_state_dict" which contain the parameters of the last Conv3D layer of the model
# the wrong thing is the number of output channels of the last Conv3D layer, which should be len(label ranges).
# Load them and discard the weight of the extra channels
def fix_task_head(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path)
    head_state_dict = checkpoint["heads_state_dict"]
    task = checkpoint["task"][0]
    head_state_dict["task_heads.8.weight"] = head_state_dict["task_heads.8.weight"][
        : task.num_output_channels
    ]
    head_state_dict["task_heads.8.bias"] = head_state_dict["task_heads.8.bias"][
        : task.num_output_channels
    ]
    checkpoint["heads_state_dict"] = head_state_dict
    torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":
    dirs = [
        "/work/grana_maxillo/UNetMerging/checkpoints/TaskVectorTrainExperiment_37pgtnb8_taskvector_tf_lriac",
        "/work/grana_maxillo/UNetMerging/checkpoints/TaskVectorTrainExperiment_e7bzktrw_taskvector_tf_crownbridgeimplant",
        "/work/grana_maxillo/UNetMerging/checkpoints/TaskVectorTrainExperiment_ek90lzx4_taskvector_tf_mandible",
        "/work/grana_maxillo/UNetMerging/checkpoints/TaskVectorTrainExperiment_ff3xd0ge_taskvector_tf_pharynx",
    ]
    for dir in dirs:
        all_files_in_dir = os.listdir(dir)
        for file in all_files_in_dir:
            if file.endswith(".pth"):
                fix_task_head(os.path.join(dir, file))
