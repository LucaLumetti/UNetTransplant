from typing import Any, Dict, List

from preprocessing.dataset_class_mapping import DATASET_IDX, DATASET_ORIGINAL_LABELS


class Task:
    def __init__(self, dataset_name: str, label_ranges: List[str]):
        self.dataset_name = dataset_name
        self.dataset_idx = DATASET_IDX[dataset_name]
        self.labels_to_predict = self.label_ranges_to_list(label_ranges)
        print(self.labels_to_predict)

    def get_mapping(self):
        return self.labels_to_predict

    def get_inverse_mapping(self):
        return {v: k for k, v in self.labels_to_predict.items()}

    @property
    def num_output_channels(self):
        return len(self.labels_to_predict)

    def label_ranges_to_list(self, label_ranges: List[str]) -> Dict[int, int]:
        label_mapping = {}

        if len(label_ranges) == 1 and label_ranges[0] == "all":
            label_ranges = [
                f"{i}"
                for i in range(1, 1 + len(DATASET_ORIGINAL_LABELS[self.dataset_name]))
            ]

        for dest, label_range in enumerate(label_ranges, start=1):
            sources = []
            if "," in label_range:
                for source in label_range.split(","):
                    if "-" in source:
                        start, end = source.split("-")
                        sources.extend(list(range(int(start), int(end) + 1)))
                    else:
                        sources.append(int(source))
            elif "-" in label_range:
                start, end = label_range.split("-")
                sources.extend(list(range(int(start), int(end) + 1)))
            else:
                sources.append(int(label_range))
            for source in sources:
                label_mapping[source] = dest
        return label_mapping

    def get_label_names(
        self,
    ):
        DATASET_ORIGINAL_LABELS[self.dataset_name]
