from collections import defaultdict
from typing import Any, Dict, List, Optional

from preprocessing.dataset_class_mapping import DATASET_IDX, DATASET_ORIGINAL_LABELS


class Task:
    def __init__(
        self,
        dataset_name: str,
        label_ranges: List[str],
        task_name: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.dataset_idx = DATASET_IDX[dataset_name]
        self.task_name = task_name
        self.labels_to_predict = self.label_ranges_to_dict(label_ranges)
        self.label_ranges = self.dict_to_label_ranges(self.labels_to_predict)
        print(self.labels_to_predict)
        print(self.label_ranges)

    def get_mapping(self):
        return self.labels_to_predict

    def get_inverse_mapping(self):
        reverse_mapping = defaultdict(list)
        for k, v in self.labels_to_predict.items():
            reverse_mapping[v].append(k)
        return dict(reverse_mapping)

    @property
    def num_output_channels(self):
        return len(self.label_ranges)

    def label_ranges_to_dict(self, label_ranges: List[str]) -> Dict[int, int]:
        label_mapping = {}

        if len(label_ranges) == 1 and label_ranges[0] == "all":
            label_ranges = [
                f"{i}"
                for i in range(1, 1 + len(DATASET_ORIGINAL_LABELS[self.dataset_name]))
            ]

        label_ranges = [label_range.replace(" ", "") for label_range in label_ranges]

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

    def dict_to_label_ranges(self, label_mapping: Dict[int, int]) -> List[str]:
        label_ranges = []
        for dest in set(label_mapping.values()):
            sources = [
                str(source) for source, dest_ in label_mapping.items() if dest_ == dest
            ]
            if len(sources) == 1:
                label_ranges.append(sources[0])
            else:
                label_ranges.append(", ".join(sources))
        return label_ranges

    def get_label_names(
        self,
    ):
        DATASET_ORIGINAL_LABELS[self.dataset_name]

    def __add__(self, other):
        assert (
            self.dataset_name == other.dataset_name
        ), "Cannot add tasks from different datasets"

        # Required as i have saved a task without these attributes :(
        if not hasattr(other, "task_name"):
            other.task_name = None
        if not hasattr(self, "task_name"):
            self.task_name = None

        # Required as i have saved a task without these attributes :(
        if not hasattr(other, "label_ranges"):
            other.label_ranges = other.dict_to_label_ranges(other.labels_to_predict)
        if not hasattr(self, "label_ranges"):
            self.label_ranges = self.dict_to_label_ranges(self.labels_to_predict)

        if self.task_name is not None and other.task_name is not None:
            both_names = f"{self.task_name} and {other.task_name}"
        else:
            both_names = (
                self.task_name if self.task_name is not None else other.task_name
            )

        both_label_ranges = self.label_ranges + other.label_ranges

        tv = Task(
            dataset_name=self.dataset_name,
            label_ranges=both_label_ranges,
            task_name=both_names,
        )
        return tv
