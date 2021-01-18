# Neural Spatio-Temporal Point Processes [[arxiv](https://arxiv.org/abs/2011.04583)]

**Ricky T. Q. Chen, Brandon Amos, Maximilian Nickel**

*Abstract.* We propose a new class of parameterizations for spatio-temporal point processes which leverage Neural ODEs as a computational method and enable flexible, high-fidelity models of discrete events that are localized in continuous time and space. Central to our approach is a combination of recurrent continuous-time neural networks with two novel neural architectures, i.e., Jump and Attentive Continuous-time Normalizing Flows. This approach allows us to learn complex distributions for both the spatial and temporal domain and to condition non-trivially on the observed event history. We validate our models on data sets from a wide variety of contexts such as seismology, epidemiology, urban mobility, and neuroscience.

*TL;DR.* We explore a natural extension of deep generative modeling to time-stamped heterogeneous data sets, enabling high-fidelity models for a large variety of spatio-temporal domains.

<p align="center">
<img align="middle" src="./assets/stpp_pinwheel.gif" width="300" />
</p>

*Caption.* A Neural STPP modeling a process where each observation increases the probability of observing from the _next_ cluster in a clock-wise order. Slowly reverts back to the marginal distribution after a period of no new observations.

## Setup

Dependencies:

- PyTorch 1.6+ (https://pytorch.org/)
- torchdiffeq 0.1.0+ (`pip install torchdiffeq`)

Run at the root of this repo:
```
python setup.py build_ext --inplace
```

## Data

Code to automatically download and preprocess most data sets can also be found in the `data` folder. Simply run
```
python download_and_preprocess_<data>.py
```
where `data` is one of `citibike|covid19|earthquakes`.

The BOLD5000 dataset requires manually downloading files from their [website](https://figshare.com/articles/dataset/BOLD5000/6459449). Specifically, the files satisfying `{}_Unfilt_BOLD_CSI1_Sess-{}_Run-{}` should be unzipped and placed in the `data/bold5000/` folder.

## Training
```
# data should be one of earthquakes_jp|fmri|citibikes|covid_nj_cases|pinwheel.
data=earthquakes_jp

# train a self-exciting baseline.
python train_stpp.py --data $data --model gmm --tpp hawkes

# train a time-varying CNF.
python train_stpp.py --data $data --model tvcnf

# train a Jump CNF.
python train_stpp.py --data $data --model jumpcnf --tpp neural --solve_reverse

# train an Attentive CNF.
python train_stpp.py --data $data --model attncnf --tpp neural --l2_attn
```

See additional arguments using `python train_stpp.py --help`.

# Citations
If you find this repository helpful in your publications,
please consider citing our paper.

```
@inproceedings{chen2021neuralstpp,
title={Neural Spatio-Temporal Point Processes},
author={Ricky T. Q. Chen and Brandon Amos and Maximilian Nickel},
booktitle={International Conference on Learning Representations},
year={2021},
}
```

# Licensing
This repository is licensed under the
[CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).
