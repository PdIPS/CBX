---
title: 'CBX: Python and Julia packages for consensus-based interacting particle methods'
tags:
  - Python
  - Julia
  - Optimization
  - Sampling
authors:
  - name: Tim Roith
    orcid: 0000-0001-8440-2928
    affiliation: 1
  - name: Konstantin Riedl
    orcid: 0000-0002-2206-4334
    affiliation: "2, 3"
  - name: Claudia Totzeck
    orcid: 0000-0001-6283-7154
    affiliation: 4
  - name: Alethea Barbaro
    orcid: 0000-0001-9856-2818
    affiliation: 5  
affiliations:
 - name: Helmholtz Imaging, Deutsches Elektronen-Synchrotron DESY
   index: 1
 - name: Technical University of Munich
   index: 2
 - name: Munich Center for Machine Learning
   index: 3
 - name: University of Wuppertal
   index: 4
 - name: Technische Universiteit Delft
   index: 5  
date: 22 November, 2023
bibliography: paper.bib
---

# Summary

Addressing real-world challenges across diverse domains, including engineering, finance, machine learning, and scientific research often requires to solve a global optimization problem of the form
$$
x^* = \operatorname*{argmin}_{x\in\mathcal{X}} f(x),
$$
where, $f:\mathcal{X}\to\mathbb{R}$ is some objective function over the state space $\mathcal{X}$. While in many cases, gradient-based methods achieve state-of-the-art performance, there are various scenarios where so-called derivative-free methods are more appropriate. This can be attributed to the unavailability or difficulty in evaluating the gradient of $f$. Additionally, it might be that $f$ is non-smooth or non-convex, which also hinders the applicability of gradient-based methods. 

Numerous techniques exist for derivative-free optimization, such as random or pattern search [@friedman1947planning;@rastrigin1963convergence;@hooke1961direct], Bayesian optimization [@movckus1975bayesian] or simulated annealing [@henderson2003theory]. However, we focus on particle-based methods, specifically on consensus-based optimization (CBO) as proposed in [@pinnau2017consensus]. For an ensemble of $N$ particles $x=(x^1,\ldots, x^N)\in \mathcal{X}^N$, the update of the $i$th particle is given by
$$
x^i \gets x^i - dt\ \lambda\ (x^i - c_\alpha(x)) + \sqrt{dt}\ \sigma\ |x^i - c_\alpha(x)|\ \xi^i,
$$
where $dt, \alpha, \lambda, \sigma > 0$ are user-specified parameters and $\xi^i \sim \mathcal{N}(0,\mathrm{Id})$ are independent, identically distributed Gaussian random vectors representing noise. Moreover, $c_\alpha(x)$ denotes the consensus point, which is computed as a weighted average of the particles $x$ and serves as a momentaneous guess for the global minimizer~$x^*$.

In this paper, we introduce CBXpy and CBX.jl, providing Python and Julia implementations, respectively, for consensus-based interacting particle methods. The zoo of different variants of CBO, such as consensus-based sampling (CBS) [@carrillo2022consensus] coined the acronym CBX. The Python and Julia implementations were developed concurrently to offer a framework accessible to researchers more familiar with either language. While ensuring a similar API and core functionality in both packages, we leveraged strengths of each language and wrote idiomatic code.

![Visualization of a CBO run for the Ackley function [@ackley2012connectionist].](JOSS.png){ width=100% }

# Statement of need

As a particle method, CBO is conceptually comparable to biologically and physically inspired methods such as particle-swarm optimization (PSO) [@kennedy1995particle], simulated annealing (SA) [@henderson2003theory] or several other heuristics [@mohan2012survey;@karaboga2014comprehensive;@yang2009firefly;@bayraktar2013wind]. However, compared to these methods, CBO was designed to be amenable to a rigorous theoretical convergence analysis on the mean-field level [@carrillo2018analytical;@carrillo2021consensus;@fornasier2021consensus;@fornasier2021convergence;riedl2022leveraging;fornasier2023consensus]. 

For Python, we refer to [@duan2023pypop7] and [@scikitopt] for a collection of various derivative-free optimization strategies. A very recent implementation of Bayesian optimization is described in [@Kim2023]. PSO and SA implementations are already available in [@miranda2018pyswarms;@scikitopt;@deapJMLR2012;@pagmo2017], which are widely used by the community and provide a rich framework for the respective methods. However, adjusting these implementations to CBO is not straightforward. Furthermore, we intend to provide a lightweight and direct implementation of CBO methods, which is easy to understand and to modify. The first publicly available Python packages implementing CBO-type algorithms were given by some of the authors together with collaborators in [@Igor_CBOinPython], where CBO as in [@pinnau2017consensus] is implemented, as well as in [@Roith_polarcbo], where so-called polarized CBO [@bungert2022polarized] is implemented. The current Python package CBXpy is a complete rewrite of the latter implementation.

For Julia, PSO and SA methods are, among others, implemented in [@mogensen2018optim;@mejia2022metaheuristics;@Bergmann2022]. Similarly, one of the authors of this paper provided the first specific Julia implementation of CBO [@Bailo_consensus]. However, the current version of the package CBX.jl deviates from the previous implementation and is more closely oriented toward the Python implementation.

We summarize the motivation and main features of the packages in what follows.

- Provide a lightweight, easy-to-understand, -use and -extend implementation of CBO together with several of its variants. These include CBO with mini-batching [@carrillo2021consensus], polarized CBO [@bungert2022polarized], CBO with memory effects [@grassi2020particle;@riedl2022leveraging], and CBS [@carrillo2022consensus].
- Define structures and hierarchies to ensure a usage experience similar to optimizer classes in PyTorch [@paszke2019pytorch;@scikitopt].
- Provide numerous utilities, like performance evaluation or plotting routines tailored to CBO methods.

# Mathematical background

CBO methods use a finite number of agents $x=(x^1,\dots,x^N)$ to explore the domain and to form a global consensus about the location of the minimizer $x^*$ as time passes. They are described through a system of stochastic differential equations (SDEs), expressed in Ito's formula as
$$
dx^i_t = -\lambda\ \underbrace{(x^i_t-c_\alpha(x_t))\,dt}_{\text{consensus drift}} + \sigma\ \underbrace{D(x^i_t-c_\alpha(x_t))\,dB^i_t}_{\text{scaled diffusion}},
$$
where $\alpha,\lambda,\sigma$ are parameters as before and $((B^i_t)_{t\geq0})_{i=1,\dots,N}$ denote independent standard Brownian motions. Global information about the objective function is encoded in the consensus point $c_\alpha$, which is computed as
$$
c_\alpha(x_t) = \frac{1}{\sum_{i=1}^N \omega_\alpha(x^i_t)} \sum_{i=1}^N x^i_t\ \omega_\alpha(x^i_t), \quad\text{ with }\quad \omega_\alpha(\,\cdots\,) = \mathrm{exp}(-\alpha f(\,\cdots\,)).
$$
Each particle is driven by a drift toward the consensus (confinement) and subject to a scaled diffusion (exploration). The scaling factor of the diffusion is proportional to the distance of the particle from the consensus point. Hence, whenever a particle's position and the location of the weighted mean coincide, the particle stops moving. Concerning the analysis of the methods, the main challenge is to balance the drift and diffusion term in such a way that all particles converge to the consensus point, which is located at the globally best position of the state space. If the drift is too strong, the convergence may happen prematurely. On the other hand, if the diffusion is too strong, there may be no convergence at all. The choice of the weight function defining the mean is motivated by the Laplace principle [@dembo1998large], which ensures that the consensus point converges to the position of the particle with best objective value (assuming that this particle is unique). From a computational perspective, the method is attractive as the particle interactions scale linearly with the number of particles.

A theoretical convergence analysis is not directly conducted on the above SDE system due to its highly complex behavior, but on its macroscopic mean-field limit (infinite-particle limit) [@huang2021MFLCBO], which can be described by a nonlinear nonlocal Fokker-Planck equation [@pinnau2017consensus;@carrillo2018analytical;@carrillo2021consensus;@fornasier2021consensus;@fornasier2021convergence].
The implemented CBO code originates from a simple Euler-Maruyama time discretization of the above SDE system.
A convergence statement therefore is available in [@fornasier2021consensus;@fornasier2021convergence].
Similar analysis techniques further allowed to obtain theoretical convergence guarantees for a variety of CBO variants [@bungert2022polarized;@riedl2022leveraging;@fornasier2023consensus] as well as PSO [@qiu2022PSOconvergence].

As of now, CBX methods have been deployed in several different settings and for different purposes, such as for solving constrained optimizations [@fornasier2020consensus_sphere_convergence;@borghi2021constrained], multi-objective optimizations [@borghi2022adaptive;@klamroth2022consensus], saddle point problems [@huang2022consensus], federated learning tasks [@carrillo2023fedcbo], adversarial training [] or for sampling [@carrillo2022consensus].
In addition, recent work [@riedl2023gradient] establishes a connection of CBO to stochastic gradient descent-type methods, suggesting a more fundamental connection of theoretical interest between derivative-free and gradient-based methods.

# Features of CBXPy
![Logo of CBXPy](CBXPy.png){ width=50% }

CBXPy is designed to express common abstractions between different variants of CBO, while still allowing the freedom of customization. The term **dynamic** is used to describe an instance of a certain CBO variant. Each dynamic inherits from the base class ```CBXDynamic```, which implements the base functionality, such as

* a ```step``` method, which can then be executed in the ```solve``` method performing the optimization of a given objective funtion,
* an index-selection scheme, which allows to easily implement a mini-batched variant of custom CBO methods,
* different termination criteria, which can be used to stop the optimization process,
* different noise models, such as isotropic [@pinnau2017consensus], anisotropic [@carrillo2021consensus] or the sampling noise from [@carrillo2022consensus].

Most of the code uses basic Python functionality, where the ensemble $x$ is modeled as an array-like structure. For certain specific features, like broadcasting-behavior, array copying, and index selection, we fall back to the  ```numpy``` implementation [@harris2020array]. However, it should be noted that an adaption to PyTorch [@paszke2019pytorch;@scikitopt] is straightforward. For the computation of the consensus point, we rely on the ```logsumexp``` function from ```scipy``` [@2020SciPy-NMeth], which allows for a numerically stable and efficient evaluation of the weighted mean. Furthermore, we employ the ```matplotlib``` library [@Hunter_2007] for visualization purposes.


A simple approach for achieving parallelization is done by running multiple instances of a single dynamic in parallel. Additionally, in the python version one has the option of low-level parallelization exploiting array operations in 
```numpy```. An ensemble has the dimension $M\times N\times d$, where $M$ is the number of runs, $N$ the number of particles and $d$ the dimension of the state space.

The package is available on [GitHub](https://github.com/pdips/CBXpy) and can be installed via ```pip```, since it is released on PyPI. It is licensed under the MIT license. A documentation is available [online](https://pdips.github.io/CBXpy/).

# Features of CBX.jl
![Logo of CBX.ji](CBXjl.png){ width=50% }

CBX.jl


# Acknowledgements

TR acknowledges support from DESY (Hamburg, Germany), a member of the Helmholtz Association HGF. KR acknowledges support from the German Federal Ministry of Education and Research and the Bavarian State Ministry for Science and the Arts.

We thank the Lorentz Center in Leiden for their kind hospitality during the workshop "Purpose-driven particle systems" in Spring 2023, where this work was initiated.

# References
