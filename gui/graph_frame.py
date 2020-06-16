from collections import Sequence
from pathlib import Path
from tkinter import ttk
from tkinter import *
import queue
import threading
import itertools
import matplotlib
import time
import numpy as np
from classification.pcap_classification import PcapClassifier
from misc.data_classes import ClassifiedFlow
from misc.utils import load_model
from model.flow_pic_model import FlowPicModel

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

result_queue = queue.Queue()
COMPLETED = "completed"
TIME_INTERVAL = 60
BLOCK_INTERVAL = 2


def _blocks_to_time(block_amount):
    return 60 + (15 * (block_amount - 1))


class FlowPicGraphFrame(ttk.Frame):

    def __init__(self, parent):
        ttk.Frame.__init__(self, parent, padding=(12, 12, 12, 12))
        device = 'cuda'
        self.categories = ['browsing', 'chat', 'file_transfer', 'video', 'voip']
        model_checkpoint = '../reg_overlap_split'
        model, _, _, _ = load_model(model_checkpoint, FlowPicModel, device)
        self.classifier = PcapClassifier(model, device, num_categories=len(self.categories))

        self.max_time = 0
        self.min_time = 0
        self.flows_map = {}

        self.title = ""
        self.figure = plt.figure(figsize=(8, 8), dpi=100)
        self.figure_per_flow = plt.figure(figsize=(8, 8), dpi=100)
        self.graph = FigureCanvasTkAgg(self.figure, self)
        self.graph_per_flow = FigureCanvasTkAgg(self.figure_per_flow, self)
        self.flow_selection = ttk.Combobox(self)
        self.flow_selection.bind("<<ComboboxSelected>>", self._on_flow_select)
        self.flow_selection_label = ttk.Label(self, text="Choose specific Flow:")
        self.return_button = ttk.Button(self, text="return", command=self._on_return_click)

        self.graph._tkcanvas.grid(column=0, row=1, columnspan=3)

    @staticmethod
    def _create_graph(graph, labels, x, y_axis):
        y_axis = [flow for flow in y_axis if len(flow) > 0]
        values = list(zip(*y_axis))
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
        graph.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=len(labels))

    def _generate_predicted_graph(self, labels, x, y_axis):
        predicted_graph = self.figure.add_subplot(212)
        predicted_graph.set_title("Predicted Categories Bandwidth for " + self.title)
        self._create_graph(predicted_graph, labels, x, y_axis)

    def _generate_actual_graph(self, filepath):
        # TODO change from hardcoded values

        labels = ['browsing', 'chat', 'file_transfer']
        x = list(range(60, 360, 60))
        flow1 = [0.2, 0.26, 0.30, 0.33, 0.4]
        flow2 = [0.3, 0.24, 0.18, 0.2, 0.2]
        flow3 = [0.4, 0.4, 0.5, 0.4, 0.32]
        y_axis = [flow1, flow2, flow3]

        actual_graph = self.figure.add_subplot(211)
        actual_graph.set_title("Actual Categories Bandwidth for " + self.title)
        self._create_graph(actual_graph, labels, x, y_axis)

    def _create_combobox(self):
        self.flow_selection["values"] = list(self.flows_map.keys())

    def _on_flow_select(self, event):
        self.figure_per_flow.clear()

        graph = self.figure_per_flow.add_subplot(111)
        flow = self.flow_selection["values"][self.flow_selection.current()]
        labels, x, y_axis = self._extract_flow_values(self.flows_map[flow])

        graph.set_title("Classification for " + flow.__str__())

        self.graph._tkcanvas.grid_forget()
        self.graph_per_flow._tkcanvas.grid(column=0, row=1, columnspan=3)
        self.return_button.grid(column=0, row=0, sticky=W)
        self._create_graph(graph, labels, x, y_axis)
        graph.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), shadow=True, ncol=len(labels))
        self.graph_per_flow.draw()

    def _on_return_click(self):
        self.graph_per_flow._tkcanvas.grid_forget()
        self.graph._tkcanvas.grid(column=0, row=1, columnspan=3)
        self.flow_selection.selection_clear()
        self.return_button.grid_forget()

    def _extract_graph_values(self, flows_data):
        flows_by_start_time = [flow.flow.pcap_relative_start_time for flow_list in flows_data for flow in flow_list]
        flows_by_end_time = [flow.flow.pcap_relative_start_time + flow.flow.times[-1] for flow_list in flows_data for
                             flow in flow_list]

        self.min_time = np.min(flows_by_start_time)
        self.max_time = np.max(flows_by_end_time)
        x = list(np.arange(TIME_INTERVAL, self.max_time + TIME_INTERVAL, TIME_INTERVAL))
        flows = [[] for _ in self.categories]
        for index, flows_by_categories in enumerate(flows_data):
            start_interval = self.min_time
            for time in x:
                if time == start_interval:
                    continue
                sums = [np.sum(classified_flow.flow.sizes[((
                                                                   classified_flow.flow.times + classified_flow.flow.pcap_relative_start_time >= start_interval) & (
                                                                   classified_flow.flow.times + classified_flow.flow.pcap_relative_start_time < time))])
                        for classified_flow in flows_by_categories]
                flows[index].append(np.sum(sums))
                start_interval += TIME_INTERVAL

        flows = [(np.array(flow) / 1000) / TIME_INTERVAL for flow in flows]

        return self.categories, x, flows

    def _extract_flow_values(self, classified_flow: ClassifiedFlow):
        x = list(np.arange(BLOCK_INTERVAL, len(classified_flow.classified_blocks) + BLOCK_INTERVAL, BLOCK_INTERVAL))
        blocks = [[] for _ in self.categories]
        for block_amount in x:
            sums_per_pred = [0 for _ in self.categories]
            for index, block in enumerate(classified_flow.classified_blocks[:block_amount]):
                sizes = np.array([data[1] for data in block.block.data])
                times = np.array([data[0] for data in block.block.data])
                pred = block.pred
                if index == 0:
                    sizes = np.sum(sizes)
                else:
                    sizes = np.sum(sizes[((times >= 45) & (times <= 60))])
                sums_per_pred[pred] += sizes / 1000
            blocks = [block+[sums / _blocks_to_time(block_amount)] for block, sums in zip(blocks, sums_per_pred)]
        x = [_blocks_to_time(block_amount) for block_amount in x]

        return self.categories, x, blocks

    def classify_pcap_file(self, filepath):
        self.title = filepath.split("/")[-1]
        flows_data = self.classifier.classify_file(Path(filepath))

        labels, x, y_axis = self._extract_graph_values(flows_data)
        self.flows_map = {classified_flow.flow.five_tuple.__str__(): classified_flow for classified_flow in
                          itertools.chain.from_iterable(flows_data)}

        self._create_combobox()
        self._generate_predicted_graph(labels, x, y_axis)
        self._generate_actual_graph(filepath)
        result_queue.put(COMPLETED)

    def clear_graphs(self):
        self._on_return_click()
        self.figure.clear()
        self.graph.draw()
        # self.figure_per_flow.clear()
        # self.graph_per_flow.draw()
        self.flow_selection_label.grid_forget()
        self.flow_selection.grid_forget()

    def draw_graphs(self):
        self.figure.subplots_adjust(hspace=0.5)
        self.graph.draw()
        self.flow_selection_label.grid(column=1, row=2, sticky=W + E + N + S)
        self.flow_selection.grid(column=1, row=3, sticky=W + E + N + S)
