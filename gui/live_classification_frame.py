from collections import Sequence
from pathlib import Path
from tkinter import ttk
from tkinter import *
import queue
import threading
import itertools
from itertools import groupby
from typing import List
import matplotlib
import time
import numpy as np
from sklearn.metrics import f1_score
import torch
from classification.clasifiers import PcapClassifier, FlowCsvClassifier
from misc.constants import BLOCK_INTERVAL, BYTES_IN_KB
from misc.data_classes import ClassifiedFlow
from misc.output import Progress
from misc.utils import load_model
from model.flow_pic_model import FlowPicModel

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

blocks_queue = queue.Queue()

class LiveClassificationFrame(ttk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)

        self.max_time = 0
        self.min_time = 0
        self.flows_map = {}

        # Model initialization
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classifier_categories = ['browsing', 'chat', 'file_transfer', 'video', 'voip']
        self.all_categories = self.classifier_categories + ['unknown']
        model_checkpoint = '../model'
        model, _, _, _ = load_model(model_checkpoint, FlowPicModel, device)
        # TODO add correct model

        # Graph initialization
        self.graph_frame = ttk.Frame(self, padding=(5, 5, 5, 5))
        self.graph_frame.grid(column=0, row=1, columnspan=3)

        self.figure = plt.figure(figsize=(7, 7), dpi=80)
        self.graph = FigureCanvasTkAgg(self.figure, self.graph_frame)
        self.graph._tkcanvas.grid(column=0, row=1, columnspan=3)

        self.figure_per_flow = plt.figure(figsize=(7, 7), dpi=80)
        self.graph_per_flow = FigureCanvasTkAgg(self.figure_per_flow, self.graph_frame)

        # Combobox initialization
        self.flow_selection = ttk.Combobox(self, width=50)
        self.flow_selection.bind("<<ComboboxSelected>>", self._on_flow_select)
        self.flow_selection_label = ttk.Label(self, text="Choose specific Flow:")
        self.return_button = ttk.Button(self, text="return", command=self._on_return_click)

    @staticmethod
    def _create_graph(graph, labels, x, y_axis_by_category):
        y_axis_by_category = [y_axis_per_flow for y_axis_per_flow in y_axis_by_category if len(y_axis_per_flow) > 0]
        values = list(zip(*y_axis_by_category))
        y_values = {label: [] for label in labels}
        y_values_for_fill = {label: [] for label in labels}
        for value_list in values:
            current_y = 0
            for label, value in zip(reversed(labels), reversed(value_list)):
                y_values_for_fill[label].append((current_y, current_y + value))

                current_y = current_y + value
                y_values[label].append(current_y)

        for label in labels:
            graph.plot(x, y_values[label], label=label)
            fill_values = list(zip(*y_values_for_fill[label]))
            graph.fill_between(x, fill_values[1], fill_values[0])

        graph.set_xlabel("time in seconds")
        graph.set_ylabel("Kbps")
        graph.set_xlim(x[0], x[-1])
        graph.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=len(labels))

    def _extract_flow_values(self, classified_flow: ClassifiedFlow):
        x = list(
            np.arange(0, classified_flow.flow.times[-1],
                      BLOCK_INTERVAL))
        bandwidth_per_category = [[] for _ in self.classifier_categories]

        for window_index, time_window in enumerate(x):
            min_index = max(min(window_index - 3, len(classified_flow.classified_blocks) - 1), 0)
            max_index = min(window_index, len(classified_flow.classified_blocks) - 1)
            times = np.array(classified_flow.flow.times)
            sizes = np.array(classified_flow.flow.sizes)
            size_in_bits = np.sum(sizes[((time_window <= times) & (times < time_window + BLOCK_INTERVAL))]) * 8
            prob_sum = np.array([0.0] * len(self.classifier_categories))
            for block in classified_flow.classified_blocks[min_index:max_index + 1]:
                prob_sum += block.probabilities
            prob_sum /= (max_index + 1) - min_index
            for index, bandwidth_list in enumerate(bandwidth_per_category):
                bandwidth_list.append(prob_sum[index] * ((size_in_bits / BYTES_IN_KB) / BLOCK_INTERVAL))

        return self.classifier_categories, x, bandwidth_per_category

    def _on_flow_select(self, event):
        self.figure_per_flow.clear()

        graph = self.figure_per_flow.add_subplot(111)
        key = str(self.flow_selection.get()).split("(")[0]
        flow = self.flows_map[key]
        labels, x, y_axis = self._extract_flow_values(flow)

        graph.set_title("Classification for " + str(flow.flow.five_tuple))

        self.graph._tkcanvas.grid_forget()
        self.graph_per_flow._tkcanvas.grid(column=0, row=1, columnspan=3)
        self.return_button.grid(column=0, row=0, sticky=W)
        self._create_graph(graph, labels, x, y_axis)
        graph.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=len(labels))
        self.figure_per_flow.tight_layout()
        self.graph_per_flow.draw()

    def _on_return_click(self):
        self.graph_per_flow._tkcanvas.grid_forget()
        self.graph._tkcanvas.grid(column=0, row=1, columnspan=3)
        self.flow_selection.set('')
        self.return_button.grid_forget()

    def begin_live_classification(self, interface):
        pass

    def stop(self):
        pass

    def clear(self):
        self._on_return_click()
        self.figure.clear()
        self.graph.draw()
        self.flow_selection_label.grid_forget()
        self.flow_selection.grid_forget()
