from tkinter import ttk
from tkinter import *
import queue
import threading
import matplotlib
import time
import numpy as np
import torch
from classification.clasifiers import Classifier
from flowpic_dataset.dataset import BlocksDataSet
from misc.constants import BLOCK_DURATION, BLOCK_INTERVAL, BYTES_IN_KB, BYTES_IN_BITS
from misc.data_classes import ClassifiedFlow, ClassifiedBlock
from misc.output import Progress
from misc.utils import load_model, chain_list_of_tuples, get_block_sizes_array, get_block_times_array
from model.flow_pic_model import FlowPicModel
from live_classification.live_capture import LiveCaptureProvider

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

action_queue = queue.Queue()
LIVE_CAPTURE_QUEUE_CHECK_INTERVAL = 1000


class LiveClassificationFrame(ttk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)

        self.max_time = 0
        self.min_time = 0
        self.flows_map = {}
        self.blocks_in_intervals = []

        # Model initialization
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classifier_categories = ['browsing', 'chat', 'file_transfer', 'video', 'voip']
        self.all_categories = self.classifier_categories + ['unknown']
        model_checkpoint = '../model'
        model, _, _, _ = load_model(model_checkpoint, FlowPicModel, device)
        self.classifier = Classifier(model, device)

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

    @staticmethod
    def _get_classified_block_mask(classified_block: ClassifiedBlock, time_interval):
        return (get_block_times_array(classified_block) > time_interval - BLOCK_INTERVAL) & \
               (get_block_times_array(classified_block) <= time_interval)

    @staticmethod
    def _calculate_bandwidth(bandwidth, time_interval):
        return (bandwidth * BYTES_IN_BITS / BYTES_IN_KB) / time_interval

    def _extract_flow_values(self, classified_flow: ClassifiedFlow):
        # TODO need to fix so it will work with blocks
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

    def _extract_graph_values(self):
        x = list(np.arange(BLOCK_INTERVAL, BLOCK_DURATION + BLOCK_INTERVAL * (len(self.blocks_in_intervals)),
                           BLOCK_INTERVAL))
        y = [[] for _ in self.all_categories]
        for time_interval in x:
            interval = max(int((time_interval - BLOCK_DURATION) / BLOCK_INTERVAL), 0)
            print(f"time: {time_interval}, interval: {interval}")
            for index, blocks_list in enumerate(self.blocks_in_intervals[interval]):
                bandwidth = np.sum(chain_list_of_tuples(
                    [get_block_sizes_array(classified_block)[
                         self._get_classified_block_mask(classified_block, time_interval)]
                     for classified_block in blocks_list]))

                bandwidth = self._calculate_bandwidth(bandwidth, BLOCK_INTERVAL)
                y[index].append(bandwidth)

        # for interval_index, blocks_by_categories in enumerate(self.blocks_in_intervals):
        #     for index, blocks_list in enumerate(blocks_by_categories):
        #         if interval_index is 0:
        #             bandwidth = np.sum(chain_list_of_tuples(
        #                 [get_block_sizes_array(classified_block) for classified_block in blocks_list]))
        #             bandwidth = self._calculate_bandwidth(bandwidth, BLOCK_DURATION)
        #             y[index].append(bandwidth)
        #         else:
        #             bandwidth = np.sum(chain_list_of_tuples(
        #                     [get_block_sizes_array(classified_block)[self._get_classified_block_mask(classified_block)]
        #                      for classified_block in blocks_list]))
        #
        #             bandwidth = self._calculate_bandwidth(bandwidth, BLOCK_INTERVAL)
        #             y[index].append(bandwidth)

        return self.all_categories, x, y

    def _generate_predicted_graph(self, labels, x, y_axis):
        predicted_graph = self.figure.add_subplot(111)
        predicted_graph.set_title("Predicted Categories Bandwidth")
        self._create_graph(predicted_graph, labels, x, y_axis)
        self.draw_graph()

    def _check_live_capture_queue(self):
        if len(self.live_capture.queue) == 0:
            self.after(LIVE_CAPTURE_QUEUE_CHECK_INTERVAL, self._check_live_capture_queue)
            return

        batch = self.live_capture.queue.pop()
        if len(batch) == 0:
            self.after(LIVE_CAPTURE_QUEUE_CHECK_INTERVAL, self._check_live_capture_queue)
            return

        flows, blocks = zip(*batch)
        blocks_ds = BlocksDataSet.from_blocks(blocks)
        _, classified_blocks = self.classifier.classify_dataset(blocks_ds)

        results = zip(flows, classified_blocks)
        for (flow, block) in results:
            if self.flows_map.__contains__(flow):
                self.flows_map[flow].append(block)
            else:
                self.flows_map[flow] = [block]

        blocks_by_categories = [[] for _ in self.all_categories]
        for classified_block in classified_blocks:
            blocks_by_categories[classified_block.pred].append(classified_block)

        self.blocks_in_intervals.append(blocks_by_categories)
        labels, x, y_axis = self._extract_graph_values()

        self.figure.clear()
        self._generate_predicted_graph(labels, x, y_axis)

        self.after(LIVE_CAPTURE_QUEUE_CHECK_INTERVAL, self._check_live_capture_queue)

    def begin_live_classification(self, interfaces, save_to_file):
        self.live_capture = LiveCaptureProvider(interfaces)
        self.live_capture_thread = threading.Thread(target=lambda: self.live_capture.start_capture())
        self.live_capture_thread.start()
        self.after(LIVE_CAPTURE_QUEUE_CHECK_INTERVAL, self._check_live_capture_queue)

    def draw_graph(self):
        self.figure.subplots_adjust(hspace=0.5)
        self.graph.draw()

    def stop(self):
        self.flow_selection["values"] = list(
            map(lambda item: f'{item[0]}', self.flows_map.items()))
        self.flow_selection_label.grid(column=1, row=2, sticky=W + E + N + S)
        self.flow_selection.grid(column=1, row=3, sticky=W + E + N + S)

    def clear(self):
        self._on_return_click()
        self.figure.clear()
        self.graph.draw()
        self.flow_selection_label.grid_forget()
        self.flow_selection.grid_forget()
