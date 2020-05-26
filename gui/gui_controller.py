from tkinter import ttk
from tkinter import *
from tkinter import filedialog
import matplotlib
import threading
import queue

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

LARGE_FONT = ("Verdana", 12)
result_queue = queue.Queue()


class GuiController(Tk):

    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)

        Tk.wm_title(self, "Traffic Classification")

        container = ttk.Frame(self)
        container.grid(column=0, row=0, sticky=(N, S, E, W))

        self.frames = {}

        for f in (StartPage, PcapClassificationPage, LiveClassificationPage):
            frame = f(container, self)
            self.frames[f] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class LiveClassificationPage(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent, padding=(12, 12, 12, 12))
        title_label = ttk.Label(self, text="Live Classification", font=LARGE_FONT)
        title_label.grid(column=2, row=0, columnspan=2)

        live_button = ttk.Button(self, text="Start Live Capturing", width=24)
        back_button = ttk.Button(self, width=24, text="Back",
                                 command=lambda: controller.show_frame(StartPage))
        live_button.grid(column=4, row=6, columnspan=2)
        back_button.grid(column=0, row=6, columnspan=2)


class PcapClassificationPage(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent, padding=(12, 12, 12, 12))

        self.pcap_file_path = None
        self.pcap_file_name = ""
        self.pcap_file_label_variable = StringVar(self)
        self.pcap_values = []
        self.pcap_lables = []
        self.controller = controller
        self.slider_graph = None

        progress_bar_frame = ttk.Frame(self, padding=(5, 5, 5, 5), width=200)
        self.progress_bar = ttk.Progressbar(progress_bar_frame, orient=HORIZONTAL, length=200,
                                            mode='indeterminate')
        self.pcap_label = ttk.Label(progress_bar_frame, font=LARGE_FONT, anchor="center",
                                    textvariable=self.pcap_file_label_variable)

        self.figure = plt.figure(figsize=(8, 8), dpi=100)
        self.graph = FigureCanvasTkAgg(self.figure, self)
        self.graph._tkcanvas.grid(column=1, row=1, columnspan=4, rowspan=4)

        self.slider = Scale(self, length=200, from_=60, to=300, orient=HORIZONTAL, resolution=60,
                            command=self.on_slider_change)
        title_label = ttk.Label(self, font=LARGE_FONT, anchor="center", text="Pcap File Classification")

        pcap_button = ttk.Button(self, text="Upload pcap file", width=24, command=self.upload_pcap_file)
        back_button = ttk.Button(self, width=24, text="Back",
                                 command=lambda: self.on_back_button_click())

        pcap_button.grid(column=4, row=8, columnspan=2)
        back_button.grid(column=0, row=8, columnspan=2)
        title_label.grid(column=2, row=0, columnspan=2)
        progress_bar_frame.grid(column=2, row=6, columnspan=2)

    def upload_pcap_file(self):
        self.figure.clear()
        # Open pcap file and save its location and name
        self.pcap_file_path = filedialog.askopenfilename()
        self.pcap_file_name = self.pcap_file_path.split("/")[-1]
        self.pcap_file_label_variable.set("Classifying " + self.pcap_file_name)
        # Show Classifying animation
        self.pcap_label.grid(column=2, row=6, columnspan=2)
        self.progress_bar.grid(column=2, row=7, columnspan=2)
        self.progress_bar.start(15)
        # Begin classifying process
        threading.Thread(target=lambda: self.classify_pcap_file()).start()

        self.master.after(100, self.check_classify_progress)

    def check_classify_progress(self):
        try:
            labels, x, y_axis = result_queue.get()
            self.progress_bar.grid_forget()
            self.pcap_label.grid_forget()
            self.slider.grid(column=2, row=5, columnspan=2)
            self.pcap_values = list(zip(*y_axis))
            self.pcap_lables = labels
            self.create_graph(labels, x, y_axis)
        except queue.Empty:
            pass

    def on_slider_change(self, value):
        if self.slider_graph is not None:
            x_axis = list(range(60, 360, 60))
            self.slider_graph.clear()
            y = self.pcap_values[x_axis.index(int(value))]
            for label, value in zip(self.pcap_lables, y):
                self.slider_graph.barh(label, value)
            self.slider_graph.set_xlim(0, 1)
            for i, v in enumerate(y):
                self.slider_graph.text(v, i, str(v), fontweight='bold')

            self.graph.draw()

    def create_graph(self, labels, x, y_axis):
        p = self.figure.add_subplot(211)
        self.figure.suptitle(self.pcap_file_name)
        for label, y in zip(labels, y_axis):
            p.plot(x, y, label=label)
        p.legend()

        self.slider_graph = self.figure.add_subplot(212)
        y = self.pcap_values[0]
        for label, value in zip(labels, y):
            self.slider_graph.barh(label, value)
        self.slider_graph.set_xlim(0, 1)
        for i, v in enumerate(y):
            self.slider_graph.text(v, i, str(v), fontweight='bold')

        self.graph.draw()

    def classify_pcap_file(self):
        # TODO change to actual classification
        labels = ["Video", "Chat", "VoIP"]
        x = list(range(60, 360, 60))
        print(x)
        flow1 = [0.2, 0.26, 0.30, 0.33, 0.4]
        flow2 = [0.3, 0.24, 0.18, 0.2, 0.2]
        flow3 = [0.4, 0.4, 0.5, 0.4, 0.32]
        y_axis = [flow1, flow2, flow3]
        result_queue.put((labels, x, y_axis))

    def on_back_button_click(self):
        self.progress_bar.stop()
        self.progress_bar.grid_forget()
        self.pcap_label.grid_forget()
        self.slider.grid_forget()
        self.figure.clear()
        self.graph.draw()
        self.controller.show_frame(StartPage)


class StartPage(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent, padding=(12, 12, 12, 12))
        title_label = ttk.Label(self, text="Traffic Classification", font=LARGE_FONT)
        title_label.pack(pady=10, padx=10)

        pcap_class_button = ttk.Button(self, text="Pcap file Classification",
                                       command=lambda: controller.show_frame(PcapClassificationPage))
        pcap_class_button.pack(side=RIGHT)

        live_class_button = ttk.Button(self, text="Live Classification",
                                       command=lambda: controller.show_frame(LiveClassificationPage))
        live_class_button.pack(side=LEFT)


if __name__ == '__main__':
    app = GuiController()
    app.mainloop()
