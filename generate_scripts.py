import numpy
from pathlib import Path
import sys

sys.path.append("../")
sys.path.append("./")

if __name__ == '__main__':
    lr_list = numpy.logspace(start=-3, stop=-1, num=3)
    reg_list = numpy.logspace(start=-3, stop=-2, num=2)
    reg_list.put(0, 0)
    print(lr_list)
    print(reg_list)
    bs_list = [64, 128, 256]
    path = "./scripts/hyper_param_scripts"
    Path(path).mkdir(parents=True, exist_ok=True)
    for lr in lr_list:
        for reg in reg_list:
            for bs in bs_list:
                filename: str = "nvpn_lr-" + str(lr) + "_reg-" + str(reg) + "_bs-" + str(bs)
                f = open(path + "\\" + filename + ".sh", "w+")
                text: str = "#!/usr/bin/env bash\n\n" \
                            "out_dir=hyper_param_out\nmkdir -p $out_dir\n" \
                            "log_path=$out_dir/" + filename + "_out.log\n\n python expiraments/split_experiment.py " \
                                                              "--data-dir data_reg_overlap_split " \
                                                              "--out-dir $out_dir " \
                                                              "--epochs 60 " \
                                                              "--lr " + str(lr) + " --reg " + str(reg) + \
                            " --bs-train " + str(bs) + " --checkpoints " + filename + \
                            " --load-checkpoint 0 " \
                            " --checkpoint-every 30 | tee $log_path "
                f.write(text)
                f.close()
