# Deep-Learning-based-Traffic-Classification

## 1. Introduction
**Deep Learning-Based Traffic Classification** is a project that was done under the course Project in Computer Communication (236340) in The Technion, in collaboration with [Rafael](https://www.rafael.co.il/). The project is based on the [FlowPic paper](https://ieeexplore.ieee.org/abstract/document/8845315) by Mr. Tal Shapira and Professor Yuval Shavitt. In this project, we created a classifier for network traffic. The classifier provides a classification to categories on a per-flow basis. Unlike traditional solutions for these  types of classifiers that rely on a flow's 4-tuple or DPI: we monitor packets' **arrival times and packet sizes only** (henceforth refered to as "samples"), and provide a robust classification for even encrypted traffic, with relatively small reliance on heavy resources. Our classifier is based on a CNN (Convolutional Neural Network) architecture, and thus - we transform collected network traffic into **FlowPics**: "pictures" that represent the arrival times and packet sizes of a flow over a fixed period of time. Along with building the classifier itself, the project also entails traffic capturing & parsing, live monitoring for online classification and an interactive user interface.

### Terminology
* Sample - a 2-tuple of (`packet_arrival_time`, `packet_size`); where `packet_arrival_time` is the inter-arrival time of the packet and `packet_size` excludes app/transport/IP headers.
* Stream - an unfiltered and unprocessed contiguous sequence of samples\packets of a 5-tuple connection (ip_src, port_src, ip_dst, port_dst, protocol).
* Flow - all the samples in a specific 5-tuple connection where the `packet_size` is lesser then 1500 and `packet_arrival_time` is normalized to start from 0 (class: `Flow`).
* Block - all the samples of a specific flow during a time interval of 60 seconds where `packet_arrival_time` is normalized to start from 0 (class: `Block`)
* FlowPic - a 2D histogram which is constructed from a specific block, where X-axis is time, Y-axis is packet size
    * by default the dimensions are 1500x1500 (60 seconds time normalized to 1500; packet size: MTU is 1500 bytes)
    * this is the only valid input for the model

## 2. Requirements And Set-Up
### Wireshark
See guide for setting up wireshark in the drive under Networking Project/Wireshark_Setup

### Conda
TODO

## 3. Training on The Technion Servers
[link1](https://vistalab-technion.github.io/cs236781/assignments/hpc-servers) and [link2](https://vistalab-technion.github.io/cs236781/assignments/getting-started) 
contain great explanations on how to use the servers.  
Here, we will provide an explanation on how we interacted with the servers but it doesn't mean that there are no
other options and we recommend to read the explanations in the links.  

### Connecting
let user@campus.technion.ac.il be our Technion mail user  
First of all connect to the csl3 server with your user by executing: `ssh -X user@csl3.cs.technion.ac.il` (password is the same as to your email).      
Now, connect from csl3 to the rishon server by executing: `ssh user@rishon.cs.technion.ac.il` (password is the same as to your email).  

### Environment Set-UP
See the "Environment Set-Up" section in [link2](https://vistalab-technion.github.io/cs236781/assignments/getting-started)
for a detailed explanation on how to install and use conda on the server.

### Running Scripts
First of all create a bash script that invokes your desired python script.  
Secondly activate your environment with `conda activate <env-name>`.  
Now you are ready to run your script. 
As explained in [link1](https://vistalab-technion.github.io/cs236781/assignments/hpc-servers)
there are a few ways in which you can run the bash script but for training the model we recommend using
the `sbatch` command and for convenience you can just run the script named "run" that is in the repository.  
For example, `./run 2 1 my_script.sh output.out` means to execute my_script.sh with sbatch on node 2
with 1 GPU and 2 CPUs (twice the number of GPUs) where the stdout will be written to output.out.  
Additionally, you can see currently running jobs and their ids with the `squeue` command, and if you want to cancel
a current running job of yours, you can do it with `scancel <job-id>` or execute the script called "cancel"
which will find and cancel all your jobs. Another option to cancel is if you execute "run" with source i.e
`source run 2 1 my_script.sh output.out` you then can use `scancel $id` as long as you didn't exit the current shell.  
Important: "cancel" script searches for a hard coded user name so change it in the script to your user name.

### Training
To train the model you will need to run reg_categories.sh as mentioned in the previous section.  
The training takes 35 epochs (around 2.5 hours on 2 GPUs) to reach it's best results.  
The script creates a new folder named "reg_out" that is used as an output folder.  
At the end of the training the folder will contain:
* out.log - a copy of the prints to stdout
* model.py - the trained model (can be passed as input to function load_model())
* acc.png - graph of accuracy as function of epoch number
* f1.png - graph of f1_score as function of epoch number
* loss.png - graph of loss as function of epoch number

### Tips
* MobaXterm is a very convenient program for connecting to the servers and downloading/uploading small files.
* MobaXtrem can't copy large folders/files and it's better to use scp(secure copy protocol) for such tasks.
* Install a program which allows you to open a bash shell on windows (git bash for example)
which will allow you to use scp from your computer.
* Try changing nodes If you see that your job isn't running due to lack of resources.

## 4. Classification
We provide 3 different wrapper classes for using the model as a classifier:
* Classifier - low level api (classifying at the flow level)
* PcapClassifier - classifies flows in pcap files
* FlowCsvClassifier - classifies flows in csv files  

## 5. Credits

### 3rd Party Libraries
- Pytorch
- pyshark
- scikit-learn
- numpy
- tqdm
- matplotlib

### Mentors
We'd like to thank Mr. Itzik Ashkenazi and Mr. Aviel Glam for their wonderful guidance throughout the semester

### Special Thanks
We want to thank Tal Shapira for providing us the dataset that he and Yuval Shavitt used in their article "FlowPic: Encrypted Internet Traffic Classification is
as Easy as Image Recognition" and for helping us in the initial stages of the project.
