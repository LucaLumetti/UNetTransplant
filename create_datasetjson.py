import json
import os

dataset_name = "TotalSegmentator"
base_folder = f"./data/{dataset_name}"
# Define the paths to the folders
images_folder = f"{base_folder}/imagesTr"
labels_folder = f"{base_folder}/labelsTr"

# Define the base JSON structure
data = {
    "name": dataset_name,
    "tensorImageSize": "3D",
    "modality": {"0": "CT"},
    "labels": {
        "0": "background",
    },
    "numTraining": 0,
    "numValidation": 0,
    "numTest": 0,
    "training": [],
}

# Get the list of files in the folders
image_files = sorted(os.listdir(images_folder))
label_files = sorted(os.listdir(labels_folder))
image_files = [
    f
    for f in image_files
    if f.endswith(".nii.gz") and f"{f[:10]}.nii.gz" in label_files
]  # fix for abdomen ct 1k
label_files = [f for f in label_files if f.endswith(".nii.gz")]
# Ensure the filenames match between the two folders
if len(image_files) != len(label_files):
    raise ValueError(
        f"The number of image files and label files do not match! ({len(image_files)} vs {len(label_files)})"
    )

# Populate the training data
for img_file, lbl_file in zip(image_files, label_files):
    training_entry = {
        "image": os.path.join(images_folder, img_file),
        "label": os.path.join(labels_folder, lbl_file),
    }
    data["training"].append(training_entry)

# Update numTraining
data["numTraining"] = len(data["training"])

# Write the JSON to a file
output_file = f"{base_folder}/dataset.json"
with open(output_file, "w") as f:
    json.dump(data, f, indent=4)

print(f"JSON file '{output_file}' created with {data['numTraining']} training entries.")
