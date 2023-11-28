# Differentiable risk-coverage curve 

This repository contains the ongoing exploration of designing a loss function that allows for better selective prediction. 

This is WIP, and the code is not yet production-ready. The first target is to have a differentiable loss function based on the area-under-the-risk-coverage-curve (AURC, [Bias-Reduced Uncertainty Estimation for Deep Neural Classifiers" (ICLR 2019)](https://openreview.net/pdf?id=SJfb5jCqKm)) that can be used in a PyTorch training loop.

## Installation

The scripts require [python >= 3.8](https://www.python.org/downloads/release/python-380/) to run.
We will create a fresh [virtualenvironment](https://virtualenvwrapper.readthedocs.io/en/latest/install.html) in which to install all required packages.
```sh
mkvirtualenv -p /usr/bin/python3 diffAURC
```

Using poetry and the readily defined pyproject.toml, we will install all required packages
```sh
workon diffAURC 
pip3 install poetry
poetry install
```

## Organization

The repository is organized as follows:
- `data/` contains some data used for the unit tests (CIFAR-100 validation/test logits)
- `paper/` contains the LateX documenting the mathematical derivation of the differentiable AURC loss function
- `src/` contains the source code
    - [AURC_implementations](src/AURC_implementations.py) holds all different implementations of the AURC metric from the literature and our own derivations/approximations
    - [AURC loss](src/AURC_loss.py) contains the implementation of the differentiable AURC loss function with the additional alphas approximation
    - [Tests](src/test_differentiability_AURC.py) contains synthetic and real data tests to check the difference in AURC metric implementations, the differentiability of the AURC loss function, and the effect of batch size on the statistical approximation quality 
    - [Metrics](src/metrics.py) contains some additional metrics to evaluate the quality of a probabilistic classifier (BS, NLL, ECE, MCE, AURC, ...)
