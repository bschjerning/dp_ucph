# Part III: Structural estimation of models with continuous (and discrete) choices
### Content
In this series of lectures on dynamic discrete-continuous choice models we will consider solution methods based on the Euler equation. We start with a simple consumption savings model that we solve using VFI, time iterations and the The Endogenous Grid Method (EGM). We the move on to models that combine discrete and continuous choices and show how DC-EGM can be used to solve a class of Discrete-Continuous Dynamic Choice Model - fast and accurately. Finally, we illustrate how these solution methods can be used as the inner loop of the nested algorithms that allows structural estimation of such models (using for example MLE, SMD, MSM). 

1. The Euler Equation, Time iterations and EGM [[Slides]](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/1_euler_egm.ipynb) 
1. Discrete-Continuous Choice Models  (DC-EGM) [[Slides]](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/2_dcegm.pdf)
1. More on Stuctural Estimation: [[Slides]](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/3_struct_est.pdf)         
1. Empirical application of DC-EGM (Iskhakov and Keane, JoE 2021): [[Slides]](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/4_aupens_dc_egm.pdf)              



###  DC-EGM: Fedor Iskhakov's MATLAB code for solving lifecycle models of consumption and savings, with additional discrete choices.
This folder [dc_egm_matlab](https://github.com/bschjerning/dp_ucph/tree/main/3_discrete_continuous_choice/dc_egm_matlab) contains a copy of the code developed by Fedor Ishkakov, Australian National University, that implements the EGM and DC-EGM algorithms. I will use the code for illustrations during the lecture on DC-EGM. 

The original code is is available at the dcegm repository: 

**[github.com/fediskhakov/dcegm](github.com/fediskhakov/dcegm)**

Below is copy of a part of the [README.md](https://github.com/fediskhakov/dcegm/blob/master/README.md) file: 
This repository contains Matlab implementation of EGM and DC-EGM algorithms for solving dynamic stochastic life-cycle models of consumption and savings, with additional discrete choices.

Three models are solved using these methods:
- Phelps model of consumption and savings with stochastic returns (and credit constraints) (EGM)
- Deaton model of consumption and savings with idiosyncratic wage shocks and credit constraints (EGM)
- Model of consumption, saving and retirement decisions with idiosyncratic wage shocks, credit constraints and absorbing retirement (DC-EGM)
- The code also contains the polyline.m class which presents a set of tools for working with linearly interpolated functions, including the upper envelope algorithm. The code also contains the easy start implementation of EGM algorithm in just 13 lines of code.

### Key References
- Christopher D. Carroll "[The method of endogenous gridpoints for solving dynamic stochastic optimization problems](http://www.sciencedirect.com/science/article/pii/S0165176505003368)" (Economics Letters, 2006)
- Iskhakov, Jorgensen, Rust and Schjerning "[The Endogenous Grid Method for Discrete-Continuous Dynamic Choice Models with (or without) Taste Shocks](http://onlinelibrary.wiley.com/doi/10.3982/QE643/full)" (Quantitative Economics, 2017)
- Iskhakov and Keane "[Effects of taxes and safety net pensions on life-cycle labor supply, savings and human capital: The case of Australia](https://doi.org/10.1016/j.jeconom.2020.01.023)" (Journal of Econometrics, 2021)
