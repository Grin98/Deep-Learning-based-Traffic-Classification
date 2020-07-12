# Deep-Learning-based-Traffic-Classification

## 1. Introduction

**Deep Learning-Based Traffic Classification** is a project that was done under the course Project in Computer Communication (236340) in The Technion, in collaboration with [Rafael](https://www.rafael.co.il/). The project is based on the [FlowPic paper](https://ieeexplore.ieee.org/abstract/document/8845315) by Mr. Tal Shapira and Professor Yuval Shavitt. In this project, we created a classifier for network traffic. The classifier provides a classification to categories on a per-flow basis. Unlike traditional solutions for these  types of classifiers that rely on a flow's 4-tuple or DPI: we monitor packets' **arrival times and packet sizes only** (henceforth refered to as "samples"), and provide a robust classification for even encrypted traffic, with relatively small reliance on heavy resources. Our classifier is based on a CNN (Convolutional Neural Network) architecture, and thus - we transform collected network traffic into **FlowPics**: "pictures" that represent the arrival times and packet sizes of a flow over a fixed period of time. Along with building the classifier itself, the project also entails traffic capturing & parsing, live monitoring for online classification and an interactive user interface.

### Terminology

* Sample - a 2-tuple of (`packet_arrival_time`, `packet_size`); where `packet_arrival_time` is the inter-arrival time of the packet and `packet_size` excludes app/transport/IP headers
* Flow - all the packets in a specific 5-tuple connection (class: `Flow`). Here, a connection is composed of the classic 4-tuple (ip_src, port_src, ip_dst, port_dst) as well as the transport protocol
* Stream - an unfiltered contiguous sequence of samples of a flow
* Block - all the packets of a specific 5-tuple during a specific time interval (class: `Block`)
* FlowPic - a 2D histogram that is constructed from a specific block. X-axis is time, Y-axis is packet size
    * by default the dimensions are 1500x1500 (60 seconds time normalized to 1500; packet size: MTU is 1500 bytes)
    * this is the only valid input for the model


## 2. Credits

### 3rd Party Libraries

- Pytorch
- pyshark
- skit-learn
- imblearn
- tqdm
- matplotlib

### Mentors
We'd like to thank Mr. Itzik Ashkenazi and Mr. Aviel Glam for their wonderful guidance throughout the semester
