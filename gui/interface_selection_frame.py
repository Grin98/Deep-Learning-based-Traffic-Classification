import subprocess as sp
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")


class InterfaceSelectionFrame(ttk.Frame):
    def __init__(self, parent, capture_button):
        ttk.Frame.__init__(self, parent, padding=(12, 12, 12, 12))

        self.selected_interface = ""

        self.capture_button = capture_button

        self.interface_selection_label = ttk.Label(self, text="Please Choose Interface:")
        self.interface_selection_label.pack()

        self.interface_selection = ttk.Combobox(self, width=50)
        self.interface_selection.bind("<<ComboboxSelected>>", self._on_interface_select)
        self.interface_selection["values"] = self._get_net_interfaces()
        self.interface_selection.pack()

    @staticmethod
    def _get_net_interfaces():
        cmd_line = ["dumpcap", "-D"]
        output = sp.check_output(cmd_line).decode('utf-8')
        return ["Capture on all interfaces"]+[line[line.find("(") + 1:line.find(")")] for line in output.splitlines()]

    def _on_interface_select(self, event):
        self.selected_interface = self.interface_selection.get()
        self.capture_button.state(["!disabled"])

    def clear(self):
        self.interface_selection.set('')
