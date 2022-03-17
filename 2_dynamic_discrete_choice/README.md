# Part II: Structural estimation of dynamic discrete choice models

In this series of lectures on dynamic discrete choice models we will consider a variety of methods to structurally estimate dynamic discrete choice models (NFXP, MPEC, NPL, CCP type estimators). 

As a testbed for comparison we will use the canonical Bus Engine Replacement model in Rust's 1987 *Econometrica* paper:

Rust, J. (1987). Optimal replacement of GMC bus engines: An empirical model of Harold Zurcher. *Econometrica*, 999-1033. [[Link to paper]](https://editorialexpress.com/jrust/crest_lectures/zurcher.pdf)

## MATLAB code for solving the bus problem
The subfolder [zurcher_matlab](https://github.com/bschjerning/dp_ucph/tree/main/2_dynamic_discrete_choice/zurcher_matlab) includes Rust's busdata, and MATLAB code to solve and estimate this model. The [readme.md](https://github.com/bschjerning/dp_ucph/tree/main/2_dynamic_discrete_choice/zurcher_matlab#readme) file provides some (very sparse) documentation. I use the MATLAB code for some illustration in the lectures, but you will work with a Python version in the exercise classes. The MATLAB implementation incorporates the key features from Rust's original NFXP code (written in GAUSS) which is carefully described and documented in the [NFXP documentation manual](https://editorialexpress.com/jrust/nfxp.pdf) 

## Slides and videotaped lectures
Previsously recorded lectures are available at the [Lectures in Dynamic Programming playlist](https://www.youtube.com/watch?v=SbVIgzWt8So&list=PLzkJu0O0lYnEpJNYJ4Ent_qckS0OKkYYg) on Bertel Schjerning's [YouTube channel](https://www.youtube.com/user/BSchjerning). Links to slides and videos below (slides are slightly updated relative to those used in the videos)
                            
1. The Nested Fixed Point Algorithm [[Slides]](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/1_nfxp.pdf)
[[Video 1 - solving Zurcher model]](https://youtu.be/JfFCZhBYgGw) [[Video 2 - estimating Zurcher model]](https://youtu.be/YpCptgY9vzw)                                            
1. Constrained Optimization Approaches to Structural Estimation (MPEC) [[Slides]](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/2_mpec.pdf)
[[Video - MPEC vs NFXP]](https://youtu.be/1uuSTLbXyd8)

1. Sequential Estimation in Discrete Decision Problems: Nested Pseudo Likelihood (NPL) and CCP estimators [[Slides]](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/3_npl.pdf)
[[Video - NPL and CCP]](https://youtu.be/KqQaWuHvYkg)                           
1. Stationary Equilibrium: Equilibrium Trade in Automobile Markets [[Slides]](https://github.com/bschjerning/dp_ucph/blob/main/2_dynamic_discrete_choice/4_eqbtrade.pdf)
