from tkinter import ttk, IntVar
import matplotlib
from live_classification.live_capture import LiveCaptureProvider

matplotlib.use("TkAgg")


class InterfaceSelectionFrame(ttk.Frame):
    def __init__(self, parent, capture_button):
        ttk.Frame.__init__(self, parent, padding=(12, 12, 12, 12))

        self.selected_interface = ""
        self.save_to_file = IntVar()

        self.capture_button = capture_button

        self.interface_selection_label = ttk.Label(self, text="Please Choose Interface:")
        self.interface_selection_label.pack()

        self.interface_selection = ttk.Combobox(self, width=50)
        self.interface_selection.bind("<<ComboboxSelected>>", self._on_interface_select)
        self.interface_selection["values"] = ["Capture on all interfaces"] + LiveCaptureProvider.get_net_interfaces()
        self.interface_selection.pack()

        ttk.Checkbutton(self, text="Save capture to file", variable=self.save_to_file, onvalue=1, offvalue=0).pack()

    def _on_interface_select(self, event):
        self.selected_interface = self.interface_selection.get()
        if self.selected_interface == self.interface_selection["values"][0]:
            self.selected_interface = LiveCaptureProvider.get_net_interfaces()
        self.capture_button.state(["!disabled"])

    def clear(self):
        self.interface_selection.set('')
        self.save_to_file.set(0)
