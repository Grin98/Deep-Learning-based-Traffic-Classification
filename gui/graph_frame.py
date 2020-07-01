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
from misc.data_classes import ClassifiedFlow
from misc.output import Progress
from misc.utils import load_model
from model.flow_pic_model import FlowPicModel
from gui.statistics_frame import StatisticsFrame

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

result_queue = queue.Queue()
COMPLETED = "completed"
TIME_INTERVAL = 15
BLOCK_INTERVAL = 15
FLOWS_TO_CLASSIFY = 20
PCAP_KEY = "p"
CSV_KEY = "c"
BYTES_IN_KB = 1024
LARGE_FONT = ("Verdana", 12)


class FlowPicGraphFrame(ttk.Frame):

    def __init__(self, parent, progress: Progress):
        ttk.Frame.__init__(self, parent)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.categories = ['browsing', 'chat', 'file_transfer', 'video', 'voip']
        model_checkpoint = '../model'
        model, _, _, _ = load_model(model_checkpoint, FlowPicModel, device)
        self.pcap_classifier = PcapClassifier(model, device, progress=progress)
        self.csv_classifier = FlowCsvClassifier(model, device, progress=progress)

        self.max_time = 0
        self.min_time = 0
        self.flows_map = {}

        self.f1_score_frame = StatisticsFrame(self, self.categories)
        self.figure = plt.figure(figsize=(6, 6), dpi=80)
        self.figure_per_flow = plt.figure(figsize=(6, 6), dpi=80)
        self.graph = FigureCanvasTkAgg(self.figure, self)
        self.graph_per_flow = FigureCanvasTkAgg(self.figure_per_flow, self)
        self.flow_selection = ttk.Combobox(self, width=50)
        self.flow_selection.bind("<<ComboboxSelected>>", self._on_flow_select)
        self.flow_selection_label = ttk.Label(self, text="Choose specific Flow:")
        self.return_button = ttk.Button(self, text="return", command=self._on_return_click)

        self.graph._tkcanvas.grid(column=0, row=1, columnspan=3)

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

    def _generate_predicted_graph(self, labels, x, y_axis):
        predicted_graph = self.figure.add_subplot(212)
        predicted_graph.set_title("Predicted Categories Bandwidth")
        self._create_graph(predicted_graph, labels, x, y_axis)

    def _generate_actual_graph(self, files, flows_data: List[ClassifiedFlow]):
        flows_data = [flow for flow in flows_data if flow.flow.times[-1] > TIME_INTERVAL]
        flows_by_categories = [[] for _ in self.categories]
        [flows_by_categories[self.categories.index(flow.flow.app)].append(flow) for flow in flows_data]

        labels, x, y_axis = self._extract_graph_values(flows_by_categories)

        actual_graph = self.figure.add_subplot(211)
        actual_graph.set_title("Actual Categories Bandwidth")
        self._create_graph(actual_graph, labels, x, y_axis)

    def _create_combobox(self):

        self.flow_selection["values"] = list(
            map(lambda item: f'{item[0]}({self.categories[item[1].pred]})', self.flows_map.items()))

    def _on_flow_select(self, event):
        self.figure_per_flow.clear()

        graph = self.figure_per_flow.add_subplot(111)
        key = str(self.flow_selection.get()).split("(")[0]
        flow = self.flows_map[key]
        labels, x, y_axis = self._extract_flow_values(flow)

        graph.set_title("Classification for " + str(flow.flow.five_tuple))

        self.graph._tkcanvas.grid_forget()
        self.f1_score_frame.grid_forget()
        self.graph_per_flow._tkcanvas.grid(column=0, row=1, columnspan=3)
        self.return_button.grid(column=0, row=0, sticky=W)
        self._create_graph(graph, labels, x, y_axis)
        graph.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=len(labels))
        self.figure_per_flow.tight_layout()
        self.graph_per_flow.draw()

    def _on_return_click(self):
        self.graph_per_flow._tkcanvas.grid_forget()
        self.graph._tkcanvas.grid(column=0, row=1, columnspan=3)
        self.f1_score_frame.grid(column=4, row=1, columnspan=3)
        self.flow_selection.set('')
        self.return_button.grid_forget()

    def _extract_graph_values(self, flows_data):
        flows_by_start_time = [flow.flow.pcap_relative_start_time for flow_list in flows_data for flow in flow_list]
        flows_by_end_time = [flow.flow.pcap_relative_start_time + flow.flow.times[-1] for flow_list in flows_data for
                             flow in flow_list]

        self.min_time = np.min(flows_by_start_time)
        self.max_time = np.max(flows_by_end_time)
        x = list(np.arange(self.min_time, self.max_time, TIME_INTERVAL))
        flows = [[] for _ in self.categories]
        for index, flows_by_categories in enumerate(flows_data):
            for start_time in x:
                sums = [np.sum(classified_flow.flow.sizes[((
                                                                   classified_flow.flow.times + classified_flow.flow.pcap_relative_start_time >= start_time) & (
                                                                   classified_flow.flow.times + classified_flow.flow.pcap_relative_start_time < start_time + TIME_INTERVAL))])
                        for classified_flow in flows_by_categories]
                flows[index].append(np.sum(sums))

        flows = [(np.array(flow) * 8 / BYTES_IN_KB) / TIME_INTERVAL for flow in flows]

        return self.categories, x, flows

    def _extract_flow_values(self, classified_flow: ClassifiedFlow):
        x = list(
            np.arange(0, classified_flow.flow.times[-1],
                      BLOCK_INTERVAL))
        bandwidth_per_category = [[] for _ in self.categories]

        for window_index, time_window in enumerate(x):
            min_index = max(min(window_index - 3, len(classified_flow.classified_blocks) - 1), 0)
            max_index = min(window_index, len(classified_flow.classified_blocks) - 1)
            times = np.array(classified_flow.flow.times)
            sizes = np.array(classified_flow.flow.sizes)
            size_in_bits = np.sum(sizes[((time_window <= times) & (times < time_window + BLOCK_INTERVAL))]) * 8
            prob_sum = np.array([0.0] * len(self.categories))
            for block in classified_flow.classified_blocks[min_index:max_index + 1]:
                prob_sum += block.probabilities
            prob_sum /= (max_index + 1) - min_index
            for index, bandwidth_list in enumerate(bandwidth_per_category):
                bandwidth_list.append(prob_sum[index] * ((size_in_bits / BYTES_IN_KB) / BLOCK_INTERVAL))

        return self.categories, x, bandwidth_per_category

    def classify_pcap_file(self, files_list: List[Path]):
        self.title = str(list(map(lambda file: file.name, files_list)))
        files_list.sort(key=lambda file: file.suffix[1])
        files_dict = {key: list(group) for key, group in
                      groupby(files_list, key=lambda file: file.suffix[1])}
        pcap_files = files_dict.get(PCAP_KEY)
        csv_files = files_dict.get(CSV_KEY)
        flows_data = []
        csv_flows_data = []
        if csv_files is not None:
            csv_flows_data = self.csv_classifier.classify_multiple_files(csv_files)
            self._generate_actual_graph(csv_files, list(csv_flows_data))
            flows_data += csv_flows_data

        if pcap_files is not None:
            flows_data += self.pcap_classifier.classify_multiple_files(pcap_files, FLOWS_TO_CLASSIFY)

        flows_data = [flow for flow in flows_data if flow.flow.times[-1] > TIME_INTERVAL]
        flows_by_categories = [[] for _ in self.categories]
        [flows_by_categories[flow.pred].append(flow) for flow in flows_data]
        labels, x, y_axis = self._extract_graph_values(flows_by_categories)

        self.flows_map = {f'{index}: {str(classified_flow.flow.five_tuple)}': classified_flow for
                          index, classified_flow in
                          enumerate(flows_data)}
        categories_by_int = list(map(lambda category: self.categories.index(category), self.categories))
        if len(csv_flows_data) > 0:
            self.f1_score_frame.calculate_f1_score(csv_flows_data, csv_flows_data, categories_by_int)

        self._create_combobox()
        self._generate_predicted_graph(labels, x, y_axis)
        result_queue.put(COMPLETED)

    def clear_graphs(self):
        self._on_return_click()
        self.figure.clear()
        self.graph.draw()
        self.f1_score_frame.grid_forget()
        self.f1_score_frame.clear_frame()
        self.flow_selection_label.grid_forget()
        self.flow_selection.grid_forget()

    def draw_graphs(self):
        self.figure.subplots_adjust(hspace=0.5)
        self.figure.tight_layout()
        self.graph.draw()
        self.f1_score_frame.grid(column=4, row=1, columnspan=3)
        self.flow_selection_label.grid(column=1, row=2, sticky=W + E + N + S)
        self.flow_selection.grid(column=1, row=3, sticky=W + E + N + S)
