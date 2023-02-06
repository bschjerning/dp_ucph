# Exercise plan 2023

## [Part I: Theory and tools](https://github.com/bschjerning/dp_ucph/tree/main/1_theory_tools)                                                      
| Week |TA |  Exercise | Topic | New Method| Additional files |
|------|----------|-------|-------|------------------| ----|
| 7    |  Jacob & Adam |1        | [Cake-eating problem in a finite horizon](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/01_cake_eating_finite.ipynb)   | Backwards Induction       |     [Exercise_1.py](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/Exercise_1.py)              |
|      |  |2        | [Cake-eating problem in an infinite horizon](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/02_cake_eating_infinite.ipynb)| Value function iteration  |[Exercise_1.py](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/Exercise_1.py), [Exercise_2.py](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/Exercise_2.py)                  |
| 8     | Adam |3        |  [Cake-eating problem with continuous choice](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/03_cake_eating_continous.ipynb)     | Interpolation  | [Exercise_2.py](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/Exercise_2.py), [Exercise_3.py](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/Exercise_3.py)                 |
|     |    |4        | [Cake-eating problem with uncertainty](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/04_cake_eating_uncertainty.ipynb)   | Handling expectations over discrete distributions      | [Exercise_4.py](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/Exercise_4.py)                 |
|      |  |5        |  [Numerical Integration](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/05_numerical_integration.ipynb)     | Monte Carlo integration and quadrature      | [tools.py](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/tools.py)                 |
| 9    |   Jacob |6        | [Deaton model in a finite horizon](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/06_deaton_model.ipynb)   |-     |[tools.py](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/tools.py), [Exercise_6.py](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/Exercise_6.py)                |
| 10   |   Adam |7        | [Deaton model in an infinite horizon](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/07_deaton_model_infinite.ipynb)   | -      | [tools.py](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/tools.py), [Exercise_7.py](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/Exercise_7.py)                 |
|      | |8        | [Function Approximation](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/08_function_approx.ipynb)   |Cubic splines, regular polynomial regression, Chebyshev polynomimal regression       |-                  |


## [Part 2: Dynamic Discrete Choice](https://github.com/bschjerning/dp_ucph/tree/main/2_dynamic_discrete_choice)                                                      
| Week | TA |Exercise | Topic | New Method| Additional files |
|------| --- |----------|-------|-------|------------------|
| 11   |   Jacob |1         | [Rust's engine replacement model](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/01_NFXP.ipynb)      | Nested fixed point algorithm (NFXP)      |[model_zucher.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/model_zucher.py), [Solve_NFXP.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/Solve_NFXP.py), [estimate_NFXP.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/estimate_NFXP.py), [alternative_specifications_ex7.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/alternative_specifications_ex7.py), [alternative_specifications_ex9.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/alternative_specifications_ex9.py)                   |
| 12   |   Adam |2        | [Rust's engine replacement model](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/02_NFXP_simulate.ipynb)   | Simulating Data       |  [model_zucher.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/model_zucher.py), [Solve_NFXP.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/Solve_NFXP.py), [estimate_NFXP.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/estimate_NFXP.py)                |
|      |  |3        |[Rust's engine replacement model - generating demand curves](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/03_NFXP_equilibirum.ipynb)       | Finding ergodic distributions      | [model_zucher.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/model_zucher.py), [Solve_NFXP.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/Solve_NFXP.py), [estimate_NFXP.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/estimate_NFXP.py)                 |
| 13   |  Jacob | 4        | [Rust's engine replacement model - compare NFXP and MPEC](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/02_MPEC/04_MPEC.ipynb)   | MPEC      | [model_zucher.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/02_MPEC/model_zucher.py), [Solve_NFXP.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/02_MPEC/Solve_NFXP.py), [estimate_NFXP.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/02_MPEC/estimate_NFXP.py), [estimate_MPEC_exante.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/02_MPEC/Estimate_MPEC_exante.py)                 |
| 14   |  |Easter - No exercise class        |       |       |                  |
| 15   |   Adam |5        | [Rust's engine replacement model - estimation with NPL](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/03_NPL/05_NPL.ipynb)   |  Nested Pseudo Likelihood (NPL)     |  [model_zucher.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/03_NPL/model_zucher_exante.py), [Solve_NFXP.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/03_NPL/Solve_NFXP.py), [NPL_exante.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/03_NPL/NPL_exante.py)                |




## [Part 3: Discrete-Continuous Choice](https://github.com/bschjerning/dp_ucph/tree/main/3_discrete_continuous_choice)                                                      
| Week | TA |Exercise | Topic | New Method| Additional files |
|------| ---|----------|-------|-------|------------------|
|16      | Jacob | 1        | [Deaton model](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/01_time_iteration.ipynb)   | Time iterations | [Exercise_1.py](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/Exercise_1.py)      |
|      | | 2        | Deaton model - EGM (Notebook TBA) | EGM | TBA |
| 17   |   Adam |3        | [Buffer-stock model](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/03_buffer_stock_egm.ipynb)   |       | [tools.py](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/tools.py), [model_exante.py](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/model_exante.py)                 |
|      | | 4        | [Buffer-stock model - Estimation](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/03_buffer_stock_egm.ipynb)      |Estimating continuous choice with MLE and MSM       |   [tools.py](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/tools.py), [model_exante.py](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/model_exante.py), [estimate_exante.py](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/estimate_exante.py)               |
| 18   |  Jacob | 5        | [Retirement model](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/05_dc_egm.ipynb)   | DC-EGM | [tools.py](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/tools.py), [model_dc_exante.py](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/model_dc_exante.py)                 |





<!-- | 16   |  Jacob | 1        |    [Discrete-continuous choice](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/01_discrete_continous_choice.ipynb)|  Discrete-continuous choice     |[Exercise_1.py](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/Exercise_1.py)     | -->







<!-- 
| Week | Exercises |Topic | Method|
|------|---------|-----------------------------------------------------|  |
| 6    | 1 - 3   |[Exercise 1: Cake-eating problem in finite time - Method: Backwards induction](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/01_cake_eating_finite.ipynb)                          |  sadasd|
|      | 2       |[Numerical implementation of simple deterministic DP problem: The cake eating problem](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/02_cake_eating_example.ipynb)  | |
| 7    | 3       |[Deaton's model and 1d Numerical Integration](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/03_deaton_1d_integration.ipynb)    |    |
|      | 4       | [Multi-dimensional Integration: Monte Carlo and Quadrature Methods](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/04_multi_d_integration.ipynb)  <br> [Portfolio Choice Example (part 1): Integration](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/04a_portfolio_integration.ipynb)<br> [Portfolio Choice Example (part 2): Optimization](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/04b_portfolio_optimal.ipynb)|   |
| 8    | 5       | [Function Approximation + The Curse of Dimensionality](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/05_interpolation.ipynb)|  |
|      | 6       |  [Function Approximation + The Curse of Dimensionality (continued)](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/05_interpolation.ipynb) <br> [Some info on term-papers](https://github.com/bschjerning/dp_ucph/blob/main/5_term_paper/term_paoer.ipynb) |  |
 -->



<!-- 
## [Part II: Structural estimation of dynamic discrete choice models](https://github.com/bschjerning/dp_ucph/tree/main/2_dynamic_discrete_choice)      
| Week | Lecture | Day | Date      | Topic |
|------|---------|-----|-----------|------------------------------------------------------|
| 9  | 7  | wed | 01/Mar/23 | [The Nested Fixed Point Algorithm (NFXP)](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/1_nfxp.pdf)|
|    | 8  | thu | 02/Mar/23 | Continued |
| 10 | 9  | wed | 08/Mar/23 | [Constrained Optimization Approaches to Structural Estimation (MPEC)](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/2_mpec.pdf)|
|    | 10 | thu | 09/Mar/23 | [Sequential Estimation in Discrete Decision Problems: Nested Pseudo Likelihood (NPL) and CCP estimators ](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/3_npl.pdf)  |
| 11 | 11 | wed | 15/Mar/23 | [Stationary Equilibrium: Equilibrium Trade in Automobile Markets](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/4_eqbtrade.pdf)         |
|    | 12 | thu | 16/Mar/23 | Continued             |

## [Part III: Structural estimation of models with continuous (and discrete) choices](https://github.com/bschjerning/dp_ucph/tree/main/3_discrete_continuous_choice)
| Week | Lecture | Day | Date      | Topic |
|------|---------|-----|-----------|------------------------------------------------------|
| 12 | 13 | wed | 22/Mar/23 | [The Euler Equation, Time iterations and EGM](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/1_euler_egm.ipynb) |
|    | 14 | thu | 23/Mar/23 | [Discrete-Continuous Choice Models  (DC-EGM)](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/2_dcegm.pdf) | 
| 13 | 15 | wed | 29/Mar/23 | [More on Stuctural Estimation](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/3_struct_est.pdf) [simple code example of SMD]()  |
|    | 16 | thu | 30/Mar/23 | [Empirical application of DC-EGM (Iskhakov and Keane, JoE 2021)](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/4_aupens_dc_egm.pdf)|

## [Part IV: Solving and estimation of dynamic games](https://github.com/bschjerning/dp_ucph/tree/main/4_dynamic_games)	
| Week | Lecture | Day | Date      | Topic |
|------|---------|-----|-----------|------------------------------------------------------|
| 14 | 17 | wed | 05/Apr/23 | [Solving and estimating static games of incomplete information](https://github.com/bschjerning/dp_ucph/blob/main/4_dynamic_games/1_StaticGames.pdf)                          |
|    | 18 | thu | 06/Apr/23 | [Structural Estimation of Dynamic Games using MPEC, NPL and 2-step-PML](https://github.com/bschjerning/dp_ucph/blob/main/4_dynamic_games/2_DynamicGames.pdf)                  |
| 15 | 19 | wed | 12/Apr/23 | [Solving  and estimating directional dynamic games with multiple equilibria using RLS](https://github.com/bschjerning/dp_ucph/blob/main/4_dynamic_games/3_rls.pdf)  |
|    |    | thu | 13/Apr/23 | Easter: No lecture                                                                     |
| 16 | 20 | wed | 19/Apr/23 | [Solving  and estimating directional dynamic games with multiple equilibria using RLS](https://github.com/bschjerning/dp_ucph/blob/main/4_dynamic_games/4_nrls.pdf)   |
|    |    | thu | 20/Apr/23 | No lecture                                                                             |

## Part V: Work on research papers
| Week | Lecture | Day | Date      | Topic |
|------|---------|-----|-----------|------------------------------------------------------|
| 17 |  | wed | 26/Apr/23 | No lecture                           |
|    |  | thu | 27/Apr/23 | Submit project proposal              |
| 18 |  | wed | 03/May/23 | Research Workshop I (10 AM - 2 PM)   |
|    |  | thu | 04/May/23 | Research Workshop II (10 AM - 12 AM) |
| 22 |  | Mon | 05/Jun/23 | **Submit term paper (10 AM)**        | -->