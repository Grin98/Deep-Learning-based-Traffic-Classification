# Deep-Learning-based-Traffic-Classification

###Used Packages:

- Pytorch
- skit-learn
- imblearn
- tqdm
- pyshark
- matplotlib


### Terminology

* stream - a sequence of (packet_arrival_time, packet_size)
* flow - all the packets of a specific 5-tuple traffic (class Flow)
* block - all the packets of a specific 5-tuple in a specific interval (class Block)
* flow row - the format in which a flow is saved to a csv file
    * format: app|src_ip|src_port|dst_ip|dst_port|transport_protocol|start_time|length|[timestamps]|' '|[sizes]|
* block row - the format in which a block is saved to a csv file
    * format: start_time|num_packets|timestamps|sizes
* FlowPic - the 2D histogram that is constructed from a specific block
    * by default the dimensions are 1500x1500
    * the only valid input to the model
