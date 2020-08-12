#!/usr/bin/env bash

conda env create -f environment.yml
pip install torch===1.5.1 torchvision===0.6.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt