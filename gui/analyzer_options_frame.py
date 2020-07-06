import itertools
import numpy as np
import subprocess as sp
import threading
from pathlib import Path
from tkinter import ttk, filedialog
from tkinter import *
import matplotlib

from misc.constants import LABEL_LIST, LARGE_FONT
from pcap_extraction.pcap_aggregation import PcapAggregator
from pcap_extraction.pcap_analyzer import PcapAnalyzer

matplotlib.use("TkAgg")
CUSTOM_FONT = ("Verdana", 18)


class AnalyzerOptions(ttk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent, padding=(5, 5, 5, 5))
        self.title = ttk.Label(self, text='', font=CUSTOM_FONT)
        self.title.pack(side=TOP, pady=10)

        self.dominant_flow_only_frame = DominantFlowOnlyOption(self)
        self.show_analyze = FlowAnalyze(self)

    def on_dominant_click(self):
        self.show_analyze.pack_forget()
        self.dominant_flow_only_frame.pack(side=TOP, fill="both", expand=True)

    def on_analyze_click(self):
        self.dominant_flow_only_frame.clear()
        self.dominant_flow_only_frame.pack_forget()
        threading.Thread(target=self._analyze_pcap).start()

    def _analyze_pcap(self):
        self.analyzer = PcapAnalyzer(self.file)
        output = self.analyzer.analyze().split("\n")
        self.show_analyze.fill_list(output)
        self.show_analyze.pack(side=TOP, fill="both", expand=True)

    def update_file(self, filepath):
        self.file = Path(filepath)
        self.title.configure(text=self.file.name)


class DominantFlowOnlyOption(ttk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)

        self.label_options_text = ttk.Label(self, text="Please Choose Label:")
        self.label_options_text.pack()
        self.label_options = ttk.Combobox(self)
        self.label_options.bind("<<ComboboxSelected>>", self._on_label_select)
        self.label_options["values"] = LABEL_LIST
        self.label_options.pack(side=TOP)

        self.generate_button = ttk.Button(self, text="Generate CSV", command=self._on_generate_click)
        self.generate_button.state(["disabled"])
        self.generate_button.pack(side=BOTTOM, pady=20)

    def _on_generate_click(self):
        out_file = Path(filedialog.asksaveasfilename(initialfile=f'{self.master.file.stem}_{self.selected_label}.csv'))
        PcapAggregator().write_pcap_flows(out_file, self.master.file, 1, self.selected_label)

    def _on_label_select(self, event):
        self.selected_label = self.label_options.get()
        self.generate_button.state(["!disabled"])

    def clear(self):
        self.label_options.set('')
        self.generate_button.state(["disabled"])


class FlowAnalyze(ttk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        self.pcap_info = StringVar()
        ttk.Label(self, textvariable=self.pcap_info, font=LARGE_FONT).pack(side=TOP)
        list_frame = ttk.Frame(self)
        list_frame.pack(side=TOP, fill="both", expand=True)
        self.sb = ttk.Scrollbar(list_frame)
        self.sb.pack(side=RIGHT, fill=Y)
        self.flow_list = Listbox(list_frame, yscrollcommand=self.sb.set, width=150)

        self.flow_list.pack(side=LEFT, fill="both", expand=True)
        self.sb.config(command=self.flow_list.yview)

        self.entries_map = {}
        labeling_frame = ttk.Frame(self)
        labeling_frame.pack(side=TOP, fill="both", expand=True, pady=20)
        for label in LABEL_LIST:
            frame = ttk.Frame(labeling_frame)
            frame.pack(side=TOP, fill="both", expand=True)
            ttk.Label(frame, text=f'{label}:', font="Helvetica 12 bold", width=15).pack(side=LEFT, padx=5)
            ttk.Label(frame, text="indexes: ").pack(side=LEFT)
            self.entries_map[label] = ttk.Entry(frame)
            self.entries_map[label].pack(side=LEFT, padx=5)

        ttk.Button(self, text="Create CSV", width=24, command=self._on_create_csv_click).pack(side=TOP)

    def _on_create_csv_click(self):
        out_file = Path(filedialog.asksaveasfilename(initialfile=f'{self.master.file.stem}_labeled.csv'))
        indices, labels = zip(*[(self._label_indices(label, indices.get()))
                                for label, indices in self.entries_map.items()
                                if indices.get()])
        indices = list(itertools.chain.from_iterable(indices))
        labels = list(itertools.chain.from_iterable(labels))
        self.master.analyzer.write_chosen_flows(out_file, indices, labels)

    def fill_list(self, l):
        info_list = l[0].split(',')
        text = ''
        for info in info_list[1:]:
            text += f'{info.replace("}", "").replace("", "" )}\n'
        self.pcap_info.set(text)
        for index, item in enumerate(l[1:], start=1):
            self.flow_list.insert(END, f'{index}: {item}')

    @staticmethod
    def _label_indices(label: str, indices: str):
        indices = np.array(indices.split(sep=','), dtype=int)
        labels = [label] * len(indices)
        return indices, labels
