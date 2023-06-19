# PathFusion
Source code for the paper "Universal Multi-modal Entity Alignment via Iteratively Fusing Modality Similarity Paths".
![](images/framework.jpg)
Our backbone implementation (structure learning) is built on the source code from [Dual-AMN](https://github.com/MaoXinn/Dual-AMN), Thanks for the work.

## Dependency
* python 3.7+
* pytorch 1.9.0+
* ...

Dependency details can be found in requirement.txt

## Dataset
PathFusion is a unified framework for Multi-modal Entity Alignment. We use two real-world multi-modal datasets, *FB15K-DB15K* and *FB15K-YG15K* (both contains the relational, visual, and attribute modalities.) from paper "[MMKG: Multi-Modal Knowledge Graphs](https://arxiv.org/abs/1903.05485)". We also use two real-world temporal KG datasets *DICEWS* and *WY50K* (both contains the relational and temporal modalities.)


## Quick Start