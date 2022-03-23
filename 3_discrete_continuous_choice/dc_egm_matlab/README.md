# DC-EGM

This folder contains a copy of the code developed by Fedor Ishkakov, Australian National University, that implements the EGM and DC-EGM algorithms

The orriginal code is is available at the dcegm repository: 

**[github.com/fediskhakov/dcegm](github.com/fediskhakov/dcegm)**

Below is copy of a part of the [README.md](https://github.com/fediskhakov/dcegm/blob/master/README.md) file: 
This repository contains Matlab implementation of EGM and DC-EGM algorithms for solving dynamic stochastic lifecycle models of consumption and savings, with additional discrete choices.

Three models are solved using these methods:
- Phelps model of consumption and savings with stochastic returns (and credit constraints) (EGM)
- Deaton model of consumption and savings with idiosyncratic wage shocks and credit constraints (EGM)
- Model of consumption, saving and retirement decisions with idiosyncratic wage shocks, credit constraints and absorbing retirement (DC-EGM)
- The code also contains the polyline.m class which presents a set of tools for working with linearly interpolated functions, including the upper envelope algorithm. The code also contains the easy start implementation of EGM algorithm in just 13 lines of code.

**References**
Christopher D. Carroll "[The method of endogenous gridpoints for solving dynamic stochastic optimization problems] (http://www.sciencedirect.com/science/article/pii/S0165176505003368)" (Economics Letters, 2006)
Iskhakov, Jorgensen, Rust and Schjerning "[The Endogenous Grid Method for Discrete-Continuous Dynamic Choice Models with (or without) Taste Shocks] (http://onlinelibrary.wiley.com/doi/10.3982/QE643/full)" (Quantitative Economics, 2017)

