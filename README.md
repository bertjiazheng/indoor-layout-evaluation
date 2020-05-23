# Holistic 3D Vision Challenge on General Room Layout Estimation Track Evaluation Package

This code is used to evaluate [General Room Layout Estimation Track](https://competitions.codalab.org/competitions/24183) on [Holistic Scene Structures for 3D Vision Workshop](https://holistic-3d.github.io/eccv20) at ECCV 2020.

## Installation

Clone repository:
```bash
# downlload the code
git clone git@github.com:bertjiazheng/indoor-layout-evaluation.git
# install requirements
pip install -r requirements.txt
```

## Usage

In order to evaluate the layout method, execute the following command:
```bash
python evaluate.py results-folder gt-folder
```

## CodaLab Evaluation

In case you would like to know which is the evaluation script that is running in the CodaLab servers, check the [evaluate_codalab.py](evaluate_codalab.py) script.

This package runs in the following docker image: [bertjiazheng/codalab:anaconda3](https://cloud.docker.com/u/bertjiazheng/repository/docker/bertjiazheng/codalab).

## Citation

Please cite `Structured3D` in your publications if it helps your research:
```bibtex
@article{Structured3D,
  title   = {Structured3D: A Large Photo-realistic Dataset for Structured 3D Modeling},
  author  = {Jia Zheng and Junfei Zhang and Jing Li and Rui Tang and Shenghua Gao and Zihan Zhou},
  journal = {CoRR},
  volume  = {abs/1908.00222},
  year    = {2019}
}
```
