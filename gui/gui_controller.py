from datetime import datetime
import threading
import time
import tkinter
from string import Template
from tkinter import filedialog, messagebox
from timeit import Timer

from gui.analyzer_options_frame import AnalyzerOptions
from gui.graph_frame import *
from gui.interface_selection_frame import InterfaceSelectionFrame
from gui.live_classification_frame import LiveClassificationFrame
from misc.output import Progress
from misc.utils import strfdelta
from pcap_extraction.pcap_aggregation import PcapAggregator
from pcap_extraction.pcap_analyzer import PcapAnalyzer

matplotlib.use("TkAgg")

LARGE_FONT = ("Verdana", 12)
CLOCK_INTERVAL = 1000


class GuiController(Tk):

    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)

        Tk.wm_title(self, "Traffic Classification")
        self.iconbitmap("../statistics.ico")
        container = ttk.Frame(self)
        container.grid(column=0, row=0, sticky=(N, S, E, W))

        self.frames = {}

        for f in (StartPage, PcapClassificationPage, LiveClassificationPage, AnalyzerPage):
            frame = f(container, self)
            self.frames[f] = frame

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.grid(row=0, column=0, sticky="nsew")
        frame.tkraise()


class LiveClassificationPage(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent, padding=(12, 12, 12, 12))
        self.controller = controller

        self.stop_clock = False

        self.title_label = ttk.Label(self, text="Live Classification", font=LARGE_FONT)
        self.title_label.grid(column=3, row=0, sticky=W + E)

        self.live_button = ttk.Button(self, text="Start Live Capturing", width=24, command=self._on_capture_click)
        self.live_button.state(["disabled"])
        self.live_button.grid(column=4, row=6, columnspan=2)

        self.clock_label = ttk.Label(self, text="", font=LARGE_FONT)

        back_button = ttk.Button(self, width=24, text="Back", command=lambda: self.on_back_button_click())
        back_button.grid(column=0, row=6, columnspan=2)

        self.stop_button = ttk.Button(self, text="Stop", width=24, command=self._on_stop_click)

        self.interface_selection = InterfaceSelectionFrame(self, self.live_button)
        self.interface_selection.grid(column=1, row=1, columnspan=4, rowspan=2, sticky=W + E + N + S)
        self.graph = LiveClassificationFrame(self)

    def on_back_button_click(self):
        if self.stop_button.winfo_viewable():
            self._on_stop_click()
        self.graph.clear()
        self.graph.grid_forget()
        self.grid_forget()
        self.title_label.grid(column=3, row=0, sticky=W + E)
        self.interface_selection.clear()
        self.interface_selection.grid(column=1, row=1, columnspan=4, rowspan=2, sticky=W + E + N + S)
        self.live_button.state(["disabled"])
        self.controller.show_frame(StartPage)

    def _on_capture_click(self):
        interfaces = self.interface_selection.selected_interface
        save_to_file = True if self.interface_selection.save_to_file.get() == 1 else False
        self.interface_selection.clear()
        self.interface_selection.grid_forget()
        self.graph.grid(column=1, row=1, columnspan=5, rowspan=5, sticky=W + E + N + S)
        self.time_of_start = datetime.now()
        self.clock_label.grid(row=6, column=2, columnspan=2)
        self._update_clock()
        self.after(CLOCK_INTERVAL, self._update_clock)

        self.graph.begin_live_classification(interfaces, save_to_file)

        self.live_button.grid_forget()
        self.stop_button.grid(column=4, row=6, columnspan=2)

    def _update_clock(self):
        if not self.stop_clock:
            now = datetime.now() - self.time_of_start
            self.clock_label.configure(text=strfdelta(now, '%H:%M:%S'))
            self.after(1000, self._update_clock)

    def _on_stop_click(self):
        self.stop_clock = True
        self.graph.stop()
        self.stop_button.grid_forget()
        self.live_button.grid(column=4, row=6, columnspan=2)


class PcapClassificationPage(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent, padding=(12, 12, 12, 12))

        self.pcap_file_path = None
        self.pcap_file_label_variable = StringVar(self)
        self.pcap_values = []
        self.pcap_lables = []
        self.controller = controller
        self.progress = Progress()
        self.progress_text = StringVar('')

        progress_bar_frame = ttk.Frame(self, padding=(5, 5, 5, 5), width=200)
        self.progress_bar = ttk.Label(progress_bar_frame, font=LARGE_FONT, anchor="center",
                                      textvariable=self.progress_text)

        self.graph = FlowPicGraphFrame(self, self.progress)
        self.graph.grid(column=1, row=1, columnspan=5, rowspan=5, sticky=W + E + N + S)

        pcap_button = ttk.Button(self, text="Upload files", width=24, command=self.upload_pcap_file, )
        back_button = ttk.Button(self, width=24, text="Back",
                                 command=lambda: self.on_back_button_click())

        pcap_button.grid(column=4, row=8, columnspan=2)
        back_button.grid(column=0, row=8, columnspan=2)
        progress_bar_frame.grid(column=2, row=6, columnspan=2)

    def upload_pcap_file(self):
        self.graph.clear_graphs()
        # Open pcap file and save its location and name
        self.pcap_file_path = list(map(lambda file: Path(file), filedialog.askopenfilenames()))
        if len(self.pcap_file_path) == 0:
            return
        self.pcap_file_label_variable.set("Classifying files")
        # Show Classifying animation
        self.progress_bar.grid(column=2, row=7, columnspan=2)
        # Begin classifying process
        threading.Thread(target=lambda: self.graph.classify_pcap_file(self.pcap_file_path)).start()

        self.controller.after(100, self.check_classify_progress)

    def check_classify_progress(self):
        try:
            self.progress_text.set(self.progress.get())
            result = result_queue.get(0)
            if result == COMPLETED:
                self.progress_text.set('')
                self.progress_bar.grid_forget()
                self.graph.draw_graphs()
            else:
                pass
        except queue.Empty:
            self.controller.after(100, self.check_classify_progress)

    def on_back_button_click(self):
        self.progress_bar.grid_forget()
        self.graph.clear_graphs()
        self.grid_forget()
        self.controller.show_frame(StartPage)


class AnalyzerPage(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent, padding=(12, 12, 12, 12))

        self.controller = controller

        self.main_frame = ttk.Frame(self, padding=(5, 5, 5, 5), borderwidth=1, relief='sunken')
        self.main_frame.pack(side=TOP, fill="both", expand=True)

        self.title_label = ttk.Label(self.main_frame, text="Label Captured Traffic", font=LARGE_FONT)
        self.title_label.pack(side=TOP, pady=10)

        self.options_frame = AnalyzerOptions(self)

        merge_button = ttk.Button(self.main_frame, text="Merge CSVs", width=30, command=self._on_merge_click)
        merge_button.pack(side=LEFT, padx=10, pady=20)
        analyze_button = ttk.Button(self.main_frame, text="Show PCAP Flow Analysis", width=30,
                                    command=lambda: self._upload_pcap_file(self.options_frame.on_analyze_click))
        analyze_button.pack(side=LEFT, padx=10, pady=20)
        dominant_button = ttk.Button(self.main_frame, text="Label Dominant Flow in PCAP", width=30,
                                     command=lambda: self._upload_pcap_file(self.options_frame.on_dominant_click))
        dominant_button.pack(side=LEFT, padx=10, pady=20)

        self.back_button = ttk.Button(self, width=24, text="Back", command=lambda: self.on_back_button_click())
        self.back_button.pack(side=BOTTOM)

        self.return_button = ttk.Button(self, width=24, text="Return", command=lambda: self.on_return_button_click())

    def _upload_pcap_file(self, func):
        filepath = filedialog.askopenfilename()
        if filepath == '':
            return
        self.main_frame.pack_forget()
        self.options_frame.update_file(filepath)
        self.options_frame.pack(side=TOP, fill="both", expand=True)
        self.return_button.pack(side=BOTTOM, pady=20)
        self.back_button.pack_forget()
        func()

    def on_return_button_click(self):
        self.main_frame.pack(side=TOP)
        self.options_frame.pack_forget()
        self.back_button.pack(side=BOTTOM)
        self.return_button.pack_forget()

    def on_back_button_click(self):
        self.grid_forget()
        self.controller.show_frame(StartPage)

    def _on_merge_click(self):
        files = [Path(file) for file in filedialog.askopenfilenames()]
        if not files:
            return

        dir_ = files[0].parent
        out_file = dir_/f'merged_{str(time.strftime("%Y-%m-%d_%H-%M-%S"))}.csv'
        PcapAggregator().merge_csvs(out_file, files)
        messagebox.showinfo(title=f'{out_file.name}', message='created successfully')

class StartPage(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        title_label = ttk.Label(self, text="Traffic Classification", font=LARGE_FONT)
        title_label.pack(pady=10, padx=10)

        pcap_class_button = ttk.Button(self, text="File Classification",
                                       command=lambda: controller.show_frame(PcapClassificationPage), width=50)
        pcap_class_button.pack(pady=5, padx=5)

        live_class_button = ttk.Button(self, text="Live Classification",
                                       command=lambda: controller.show_frame(LiveClassificationPage), width=50)
        live_class_button.pack(pady=5, padx=5)

        analyzer_button = ttk.Button(self, text="Label Captured Traffic",
                                     command=lambda: controller.show_frame(AnalyzerPage), width=50)
        analyzer_button.pack(pady=5, padx=5)


if __name__ == '__main__':
    app = GuiController()
    app.mainloop()
