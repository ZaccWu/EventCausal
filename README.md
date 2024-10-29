# code repository for **estimating causal effects of events**.

This repository contains `PyTorch` implementations of training, evaluating, and predicting from our model. 


### Requirements

To run the code inside this repository, we highly recommend to use `Conda`, an open-source package and environment management system for `Python`. For installing `Conda`, please refer to the website https://www.anaconda.com/products/distribution. After installing `Conda`, please run the following lines in your preferred terminal to install the default environment of our code:

```
conda env create -f environment.yml
conda activate env_eventcause
```
Then, you can run your code in the conda environment named `env_eventcause`. You can also run the code in your own environment with PyTorch installed.


### File Structures

* **train.py**: Executable model training and causal effect estimation.

* **models.py**: Core Skeleton of the Model Framework.


### Replications

The code can be run with a simple demo:

```
python train.py
```

### Citation

If you use the code in this repository, please cite the following wordings:

```
Wu, Z. Code implementation for causal effect estimation for event data, accessed on October 29, 2024.
```
or the BibTeX format:

```
@article{wu2024code,
	title={Code implementation for causal effect estimation for event data},
	author={Ziyue Wu},
  note = {Accessed: 2024-10-29}
}
```
