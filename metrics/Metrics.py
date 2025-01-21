import numpy as np
import numpy.typing as npt


class Metrics:
    def compute_tp_fp_tn_fn(self, y_true: npt.NDArray, y_pred: npt.NDArray):
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)

        tp = np.sum(y_true & y_pred)
        fp = np.sum(~y_true & y_pred)
        tn = np.sum(~y_true & ~y_pred)
        fn = np.sum(y_true & ~y_pred)

        return tp, fp, tn, fn

    def compute_dice(self, tp: int, fp: int, tn: int, fn: int):
        return 2 * tp / (2 * tp + fp + fn)

    def compute_iou(self, tp: int, fp: int, tn: int, fn: int):
        return tp / (tp + fp + fn)

    def compute_sensitivity(self, tp: int, fp: int, tn: int, fn: int):
        return tp / (tp + fn)

    def compute_specificity(self, tp: int, fp: int, tn: int, fn: int):
        return tn / (tn + fp)

    def compute_precision(self, tp: int, fp: int, tn: int, fn: int):
        return tp / (tp + fp)

    def compute_recall(self, tp: int, fp: int, tn: int, fn: int):
        return tp / (tp + fn)

    def compute_f1(self, tp: int, fp: int, tn: int, fn: int):
        precision = self.compute_precision(tp, fp, tn, fn)
        recall = self.compute_recall(tp, fp, tn, fn)
        return 2 * precision * recall / (precision + recall)

    def compute_accuracy(self, tp: int, fp: int, tn: int, fn: int):
        return (tp + tn) / (tp + fp + tn + fn)

    def compute(self, y_true: npt.NDArray, y_pred: npt.NDArray):
        tp, fp, tn, fn = self.compute_tp_fp_tn_fn(y_true, y_pred)
        metrics = {
            "dice": self.compute_dice(tp, fp, tn, fn),
            "iou": self.compute_iou(tp, fp, tn, fn),
            "sensitivity": self.compute_sensitivity(tp, fp, tn, fn),
            "specificity": self.compute_specificity(tp, fp, tn, fn),
            "precision": self.compute_precision(tp, fp, tn, fn),
            "recall": self.compute_recall(tp, fp, tn, fn),
            "f1": self.compute_f1(tp, fp, tn, fn),
            "accuracy": self.compute_accuracy(tp, fp, tn, fn),
        }
        return metrics
