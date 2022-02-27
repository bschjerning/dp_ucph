**MATLAB ZURCHER CODE**
This code allows for solutions, estimation, simulation, and computation of equilibrium mileage and demand based on rust's engine replacement model Rust(Ecta, 1987). 

The MATLAB NFXP code is based on Rust original GAUSS code and documentation manual and has the following key features highlighted in the original NFXP manual

1. Solves fixed points on Bellman equations using a poly algorithm that optimally combines successive contraction iterations Newton iterations.
2. Recenter Bellman equation and logit formulas to avoid numerical overflow
3. Provides analytical gradients of Bellman operator
4. Provide analytical gradients of likelihood
5. Use BHHH (outer product of gradients as hessian approx.)

A variety of other methods proposed in the literature is also implemented (MPEC, NPL, BBL) for comparison and illustration. 

This version: August 2021

By Fedor Iskhakov, Bertel Schjerning, and John Rust

**MATLAB "run" scripts that can be called directly**
- run_busdata.m: Estimates Zurcher's model using and Rust's data. Compares NFXP and MPEC (only two-step partial MLE is available for MPEC). 

- run_errorbound.m: Program that compares convergence properties of successive approximations and Newton-Kantorovich iterations. 

- run_dem:  Computes equilibrium mileage distributions and the expected demand for bus engines over a fixed interval of time as a function of the price of new bus engines, mp.RC. 

- run_mc_nfxp: Monte Carlo experiment to asses the performance of NFXP on Rust's engine replacement model

- run_nfxp_vs_mpec.m: Compares NFXP and MPEC in Monte Carlo simulation study.

- run_sparsity.: Code that displays sparsity patterns for Hessian and Jacobian constraints (important for MPEC), the transition matrices and the derivatives of the Bellman operator. 

- run_bbl.m: Demonstrates (loosely) BBL method using Rust's engine replacement model 

- run_msm.m: Demonstrates (loosely) NFXP using Method of Simulated Moments criterion for Rust's engine replacement model. Simple 2d illustration of objective function and grid search for replacement and operating cost parameters.

**MATLAB classses**
- zurcher.m: Model class that contains all primitives of the engine replacement model of Harold Zuercher (state transition matrices, utility functions, Bellman equations, likelihood function, simulator and equilibrium calculator). 

- dpsolver.m: Solver for finding fixed point of bellman equations using a combination of Succesive Approximations (SA) and Newton-Kantorovich (NK) iterations.  

- nfxp.m: Collects functions necessary to solve and estimate Zurcher's model using the Nested Fixed point algorithm, NFXP. 

- mpec.m: Collects functions necessary to solve and estimate Zurcher's model using Mathematical Programming with Equilibrium Constrains (MPEC). 

- npl.m: Collects functions necessary to solve and estimate Zurcher's model using Nested Pseudo Likelihood (NPL). 

- msm.m: Collects functions necessary to solve and estimate Zurcher's model using Methodf of Simulated Moments (msm). 



