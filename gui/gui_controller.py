import threading
import tkinter
from tkinter import filedialog

from gui.graph_frame import *
from gui.interface_selection_frame import InterfaceSelectionFrame
from gui.live_classification_frame import LiveClassificationFrame
from misc.output import Progress

matplotlib.use("TkAgg")

LARGE_FONT = ("Verdana", 12)


class GuiController(Tk):

    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)

        Tk.wm_title(self, "Traffic Classification")
        self.iconbitmap("../statistics.ico")
        container = ttk.Frame(self)
        container.grid(column=0, row=0, sticky=(N, S, E, W))

        self.frames = {}

        for f in (StartPage, PcapClassificationPage, LiveClassificationPage):
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

        self.title_label = ttk.Label(self, text="Live Classification", font=LARGE_FONT)
        self.title_label.grid(column=3, row=0, sticky=W + E)

        self.live_button = ttk.Button(self, text="Start Live Capturing", width=24, command=self._on_capture_click)
        self.live_button.state(["disabled"])
        self.live_button.grid(column=4, row=6, columnspan=2)

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
        save_to_file = True if self.interface_selection.save_to_file == 1 else False
        self.interface_selection.clear()
        self.interface_selection.grid_forget()
        self.graph.grid(column=1, row=1, columnspan=5, rowspan=5, sticky=W + E + N + S)

        self.graph.begin_live_classification(interfaces, save_to_file)

        self.live_button.grid_forget()
        self.stop_button.grid(column=4, row=6, columnspan=2)

    def _on_stop_click(self):
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


if __name__ == '__main__':
    app = GuiController()
    app.mainloop()
