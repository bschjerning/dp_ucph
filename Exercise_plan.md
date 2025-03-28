# Exercise plan 2025

  

## [Part I: Theory and tools](https://github.com/bschjerning/dp_ucph/tree/main/1_theory_tools)                                                      

| Week |TA |  Exercise | Topic | New Method| Additional files |
|------|----------|-------|-------|------------------| ----|
| 7    |  Jacob |1        | [Cake-eating problem in a finite horizon](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/01_cake_eating_finite.ipynb)   | Backwards Induction       |     [Exercise_1.py](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/Exercise_1.py)              |
|      |  |2        | [Cake-eating problem in an infinite horizon](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/02_cake_eating_infinite.ipynb)| Value function iteration  |[Exercise_1.py](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/Exercise_1.py), [Exercise_2.py](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/Exercise_2.py)                  |
| 8     | Jacob |3        |  [Cake-eating problem with continuous choice](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/03_cake_eating_continous.ipynb)     | Interpolation  | [Exercise_2.py](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/Exercise_2.py), [Exercise_3.py](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/Exercise_3.py)                 |
|     |    |4        | [Cake-eating problem with uncertainty](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/04_cake_eating_uncertainty.ipynb)   | Handling expectations over discrete distributions      | [Exercise_4.py](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/Exercise_4.py)                 |
|      |  |5        |  [Numerical Integration](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/05_numerical_integration.ipynb)     | Monte Carlo integration and quadrature      | [tools.py](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/tools.py)                 |
| 9    |   Jacob |6        | [Deaton model in a finite horizon](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/06_deaton_model.ipynb)   |-     |[tools.py](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/tools.py), [Exercise_6.py](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/Exercise_6.py)                |
|    |    |7        | [Deaton model in an infinite horizon](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/07_deaton_model_infinite.ipynb)   | -      | [tools.py](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/tools.py), [Exercise_7.py](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/Exercise_7.py)                 |
<!-- |      | |8        | [Function Approximation](https://github.com/bschjerning/dp_ucph/blob/main/1_theory_tools/exercises/ex_ante/08_function_approx.ipynb)   |Cubic splines, regular polynomial regression, Chebyshev polynomimal regression       |-                  | -->

  
  

## [Part 2: Dynamic Discrete Choice](https://github.com/bschjerning/dp_ucph/tree/main/2_dynamic_discrete_choice)                                                      

| Week | TA | Exercise | Topic | New Method | Additional files |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 10 | Jacob | 0 | [Rust's engine replacement model](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/00_intro/00_intro_discrete_choice.ipynb) | Solving discrete choice models (actually already covered in ex 1 and 2) | [simple.zurcher.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/00_intro/simple_zurcher.py)  |
| 11 | Jacob | 1 | [Rust's engine replacement model](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/01_NFXP.ipynb) | Nested fixed point algorithm (NFXP) | [model_zucher.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/model_zucher.py), [Solve_NFXP.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/Solve_NFXP.py), [estimate_NFXP.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/estimate_NFXP.py), [alternative_specifications_ex7.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/alternative_specifications_ex7.py), [alternative_specifications_ex9.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/alternative_specifications_ex9.py) |
| 12 | Jacob | 2 | [Rust's engine replacement model](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/02_NFXP_simulate.ipynb) | Simulating Data | [model_zucher.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/model_zucher.py), [Solve_NFXP.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/Solve_NFXP.py), [estimate_NFXP.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/estimate_NFXP.py) |
| 13  | Jacob | 5 | [Rust's engine replacement model - estimation with NPL](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/03_NPL/05_NPL.ipynb) | Nested Pseudo Likelihood (NPL) | [model_zucher.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/03_NPL/model_zucher_exante.py), [Solve_NFXP.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/03_NPL/Solve_NFXP.py), [NPL_exante.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/03_NPL/NPL_exante.py) |


<!-- |  |  | 3 | [Rust's engine replacement model - generating demand curves](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/03_NFXP_equilibirum.ipynb) | Finding ergodic distributions | [model_zucher.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/model_zucher.py), [Solve_NFXP.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/Solve_NFXP.py), [estimate_NFXP.py](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/exercises/ex_ante/01_NFXP/estimate_NFXP.py) | -->



  
  
  
  

## [Part 3: Discrete-Continuous Choice](https://github.com/bschjerning/dp_ucph/tree/main/3_discrete_continuous_choice)                                                      

| Week | TA |Exercise | Topic | New Method| Additional files |
|------| ---|----------|-------|-------|------------------|
|14      | Jacob | 1        | [Deaton model](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/01_time_iteration.ipynb)   | Time iterations | [Exercise_1.py](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/Exercise_1.py)      |
|      | | 2        | [Deaton model - EGM](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/02_EGM.ipynb) | EGM | [Exercise_2.py](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/Exercise_2.py) |
| 15 |   Jacob |3        | [Buffer-stock model](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/03_buffer_stock_egm.ipynb)   | -      | [tools.py](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/tools.py), [model_exante.py](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/model_exante.py)                 |
|      | | 4        | [Buffer-stock model - Estimation](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/03_buffer_stock_egm.ipynb)      |Estimating continuous choice with MLE and MSM       |   [tools.py](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/tools.py), [model_exante.py](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/model_exante.py), [estimate_exante.py](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/estimate_exante.py)               |
| 16 | Easter  - No exercises | ---- | ---- | ---- | ---- |
| 17   |  Jacob | 5        | [Retirement model](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/05_dc_egm.ipynb)   | DC-EGM | [tools.py](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/tools.py), [model_dc_exante.py](https://github.com/bschjerning/dp_ucph/blob/main/3_discrete_continuous_choice/exercises/ex_ante/model_dc_exante.py)                 |
