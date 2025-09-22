# jmrespy
Python implementation for estimation of joint models for longitudinal and survival data with shared random effects (abbreviated JMSRE) and dynamic prediction.

The library was first designed as a python implementation of [JM](https://github.com/drizopoulos/JM) R package. Implementation of a JMSRE for multiple longitudinal markers and competing risks was then added.

A well-known issue with JMRSE is the presence of an intractable nested integral in likelihood function. Gauss-Hermite approximation of this integral is a common solution, but has an exponential complexity as the dimension of random effects grows.

This library is a python implementation of an academic research project to adress this issue by fitting JMRSE with scrambled Quasi-Monte-Carlo (QMC) approximation of the integral and maximisation of the noised likelihood using a noise-tolerant L-BFGS algorithm proposed by [Shi et al.](https://github.com/hjmshi/noise-tolerant-bfgs). This academic work is currently in revision process to be publicated.

## Scope of the library
At this date implemented JMSRE is available for the following settings :

**Baseline hazard function :**

* Weibull

**Longitudinal parameter distribution :**

* Gaussian
* Bernoulli

**Competing risks :**

* Cause-specific

**Link between longitudinal parameters and instantaneous hazard :**

* Current estimation of the longitudinal marker over time
* Slope of the longitudinal marker over time
* Both

**OS :**

* GNU/Linux
* Mac OS

## Installation
Download the source code of the repository and then run `setup.py` in a terminal, where `setup.py` is located (at the root of the folder).