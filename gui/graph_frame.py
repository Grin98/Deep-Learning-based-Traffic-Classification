from tkinter import ttk
from tkinter import *
import queue
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

result_queue = queue.Queue()
COMPLETED = "completed"


class FlowPicGraphFrame(ttk.Frame):

    def __init__(self, parent):
        ttk.Frame.__init__(self, parent, padding=(12, 12, 12, 12))

        self.title = ""
        self.figure = plt.figure(figsize=(8, 8), dpi=100)
        self.graph = FigureCanvasTkAgg(self.figure, self)
        self.graph._tkcanvas.grid(column=0, row=0)

    @staticmethod
    def __create_graph(graph, labels, x, y_axis):
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

            graph.set_xlim(x[0], x[-1])
            graph.legend()

    def __generate_predicted_graph(self, labels, x, y_axis):
        predicted_graph = self.figure.add_subplot(212)
        predicted_graph.set_title("Predicted Categories Bandwidth for " + self.title)
        self.__create_graph(predicted_graph, labels, x, y_axis)

    def __generate_actual_graph(self, filepath):
        # TODO change from hardcoded values
        labels = ["Video", "Chat", "VoIP"]
        x = list(range(60, 360, 60))
        flow1 = [0.2, 0.26, 0.30, 0.33, 0.4]
        flow2 = [0.3, 0.24, 0.18, 0.2, 0.2]
        flow3 = [0.4, 0.4, 0.5, 0.4, 0.32]
        y_axis = [flow1, flow2, flow3]

        actual_graph = self.figure.add_subplot(211)
        actual_graph.set_title("Actual Categories Bandwidth for " + self.title)
        self.__create_graph(actual_graph, labels, x, y_axis)

    def classify_pcap_file(self, filepath):
        self.title = filepath.split("/")[-1]
        # TODO change to actual classification(via the model)
        labels = ["Video", "Chat", "VoIP"]
        x = list(range(60, 360, 60))
        flow1 = [0.2, 0.26, 0.30, 0.33, 0.4]
        flow2 = [0.3, 0.24, 0.18, 0.2, 0.2]
        flow3 = [0.4, 0.4, 0.5, 0.4, 0.32]
        y_axis = [flow1, flow2, flow3]
        self.__generate_predicted_graph(labels, x, y_axis)
        self.__generate_actual_graph(filepath)
        result_queue.put(COMPLETED)

    def clear_graphs(self):
        self.figure.clear()

    def draw_graphs(self):
        self.graph.draw()
