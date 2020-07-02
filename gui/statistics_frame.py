import itertools
import sys
from tkinter import ttk, Label, StringVar
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
        self.title = Label(self, text="F1 Score:", pady=3, font='Helvetica 14 bold')
        self.f1_value_text = StringVar(self, 'Total: ')
        self.f1_value = ttk.Label(self, textvariable=self.f1_value_text, font=LARGE_FONT)

    def _f1_score(self, actual_flows_by_categories, analyzed_flows_by_categories, labels: List[int]):
        analyzed_flows_preds = np.array(list(itertools.chain.from_iterable(
            map(lambda f: [f.pred]*f.flow.num_packets, analyzed_flows_by_categories)
        )), dtype=np.int8)

        actual_flows_preds = np.array(list(itertools.chain.from_iterable(
            map(lambda f: [self.categories.index(f.flow.app)]*f.flow.num_packets, actual_flows_by_categories)
        )), dtype=np.int8)

        existing_labels: List = np.unique(actual_flows_preds).tolist()
        total_f1 = f1_score(actual_flows_preds, analyzed_flows_preds, average='weighted', labels=existing_labels)
        per_class_f1 = f1_score(actual_flows_preds, analyzed_flows_preds, average=None, labels=existing_labels)
        per_class_f1 = [per_class_f1[existing_labels.index(label)] if label in existing_labels else None for label in
                        labels]

        return total_f1, per_class_f1

    def calculate_f1_score(self, actual_flows_by_categories, analyzed_flows_by_categories, labels: List[int]):
        total_f1, per_class_f1 = self._f1_score(actual_flows_by_categories, analyzed_flows_by_categories, labels)
        total_f1 *= 100
        self.title.grid(row=0, column=0, columnspan=2)
        text = self.f1_value_text.get()
        text += f'{total_f1:.1f}%\n'
        for index, f1_score in enumerate(per_class_f1):
            if f1_score is not None:
                f1_score *= 100
                text += f'{self.categories[index]}: {f1_score:.1f}%\n'
            else:
                text += f'{self.categories[index]}: None\n'
        self.f1_value_text.set(text)
        self.f1_value.grid(row=1, column=0, columnspan=2)

    def clear_frame(self):
        self.title.pack_forget()
        self.f1_value.pack_forget()
        self.f1_value_text.set('Total: ')
