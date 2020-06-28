from tkinter import ttk, Label
from typing import List

import matplotlib
import numpy as np
from sklearn.metrics import f1_score

matplotlib.use("TkAgg")

LARGE_FONT = ("Verdana", 12)


class StatisticsFrame(ttk.Frame):
    def __init__(self, parent, categories):
        ttk.Frame.__init__(self, parent, padding=(6, 6, 6, 6), borderwidth=1, relief='sunken')
        self.categories = categories

    def _f1_score(self, actual_flows_by_categories, analyzed_flows_by_categories, labels: List[int]):
        actual_flows_preds = list(map(lambda f: f.pred, actual_flows_by_categories))
        analyzed_flows_preds = list(map(lambda f: self.categories.index(f.flow.app), analyzed_flows_by_categories))
        existing_labels: List = np.unique(actual_flows_preds).tolist()

        total_f1 = f1_score(actual_flows_preds, analyzed_flows_preds, average='weighted', labels=existing_labels)
        per_class_f1 = f1_score(actual_flows_preds, analyzed_flows_preds, average=None, labels=existing_labels)
        per_class_f1 = [per_class_f1[existing_labels.index(label)] if label in existing_labels else None for label in
                        labels]

        return total_f1, per_class_f1

    def calculate_f1_score(self, actual_flows_by_categories, analyzed_flows_by_categories, labels: List[int]):
        total_f1, per_class_f1 = self._f1_score(actual_flows_by_categories, analyzed_flows_by_categories, labels)
        total_f1 *= 100
        Label(self, text="F1 Score:", pady=3, font='Helvetica 14 bold').pack()
        ttk.Label(self, text=f'Total: {total_f1:.1f}%', font=LARGE_FONT).pack()
        for index, f1_score in enumerate(per_class_f1):
            if f1_score is not None:
                f1_score *= 100
                ttk.Label(self, text=f'{self.categories[index]}: {f1_score:.1f}%', font=LARGE_FONT).pack()
