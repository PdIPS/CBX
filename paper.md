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
date: 09 October 2023
bibliography: paper.bib
---

# Summary

Addressing real-world challenges across diverse domains, including engineering, finance, machine learning, and scientific research often requires solving a global optimization problem of the form

$$
\min_{x\in\mathcal{X}} f(x).
$$

Here, $f:\mathcal{X}\to\mathbb{R}$ is some objective function over the state space $\mathcal{X}$. While in many cases, gradient-based methods achieve state-of-the-art performance, there are various scenarios where so-called derivative free methods are more appropriate. This can be attributed to the unavailability or difficulty in evaluating the gradient of $f$. Additionally, it might be that $f$ is non-smooth or non-convex, which also hinders the applicability of gradient-based methods. 

Numerous techniques exist for derivative-free optimization, such as random or pattern search [@friedman1947planning;@rastrigin1963convergence;@hooke1961direct], Bayesian optimization [@movckus1975bayesian] or simulated annealing [@henderson2003theory]. However, we focus on particle based methods, specifically on consensus-based optimization (CBO) as proposed in [@pinnau2017consensus]. For an ensemble of particles $x=(x^1,\ldots, x^N)\in \mathcal{X}^N$ the update of the $i$ the particle is given as follows:

$$
x^i \gets x^i + dt\ \lambda\ (x^i - c_\alpha(x)) + \sigma\ |x^i - c(x)|\ \xi^i,
$$

where $dt, \lambda, \alpha, \sigma > 0$ are parameters, $c(x)$ is the consensus point, and $\xi^i \sim \mathcal{N}(0,1)$ represents noise.

In this paper we introduce CBXpy and CBX.jl, providing Python and Julia implementations, respectively, for consensus-based interacting particle methods. The zoo of different variants of CBO, such as consensus-based sampling (CBS) [@carrillo2022consensus] coined the acronym CBX. The Python and Julia implementations were developed concurrently to offer a framework accessible for researchers more familiar with either language. While ensuring a similar API and core functionality in both packages, we leveraged strengths of each language and wrote idiomatic code.

![Visualization of a CBO run for the Ackley function [@ackley2012connectionist].](JOSS.png){ width=50% }

# Statement of need

As a particle method, CBO is conceptually comparable to biologically and physically inspired methods such as particle-swarm optimization (PSO) [@kennedy1995particle], simulated annealing (SA) [@henderson2003theory] or several other heuristics [@mohan2012survey;@karaboga2014comprehensive;@yang2009firefly;@bayraktar2013wind]. However, compared to these methods, CBO was designed to be amenable to a rigorous theoretical convergence analysis on the mean-field level [@carrillo2018analytical;@carrillo2021consensus;@fornasier2021consensus;@fornasier2021convergence]. 

For Python, various derivative-free optimization strategies, we refer to [@duan2023pypop7] and[@scikitopt]. A very recent implementation of Bayesian optimization is described in [@Kim2023]. PSO and SA implementations are already available [@miranda2018pyswarms;@scikitopt;@deapJMLR2012;@pagmo2017], which are widely used in the community and provide a rich framework for the respective methods. However, adjusting these implementations to CBO is not straightforward. Furthermore, in this project, we want to provide a lightweight and direct implementation of CBO methods, which are easy to understand and to modify. The first publicly available Python packages implementing CBO-type algorithms were given by some of the authors together with collaborators in [@Igor_CBOinPython], where CBO as in [@pinnau2017consensus] is implemented, as well as in [@Roith_polarcbo], where so-called polarized CBO [@bungert2022polarized] is implemented. The current Python package is a complete rewrite of the latter implementation.

For Julia, PSO and SA methods are, among others, implemented in [@mogensen2018optim;@mejia2022metaheuristics;@Bergmann2022]. Similarly, one of the authors provided the first specific Julia implementation of CBO [@Bailo_consensus]. However, the current version of the package deviates from the previous implementation and is more closely oriented toward the Python implementation.

We summarize the motivation and main features of the packages in what follows.

- Provide a lightweight, easy-to-understand, -use and -extend implementation of CBO together with several of its variants. These include CBO with mini-batching [@carrillo2021consensus], polarized CBO [@bungert2022polarized], CBO with memory effects [@grassi2020particle;@riedl2022leveraging], and CBS [@carrillo2022consensus].
- Define structures and hierarchies to ensure a usage similar to optimizer classes in PyTorch [@paszke2019pytorch;scikitopt]
- Provide numerous utilities, like performance evaluation or plotting routines tailored to CBO methods

# Mathematical background


CBO methods use a finite number of agents $x=(x^1,\dots,x^N)$ to explore the domain and to form a global consensus about the location of the minimizer $x^*$ as time passes. They are described through a system of stochastic differential equations (SDEs), expressed in It\^o's form as
$$
dx^i_t = -\lambda\ \underbrace{(x^i_t-c_\alpha(x))}_{\text{drift}} dt + \sigma\ \underbrace{D(x^i_t-c_\alpha(x))\ dB^i_t}_{\text{scaled diffusion}},
$$
where $\alpha,\lambda$ and $\sigma$ are user-specified parameters and $((B^i_t)_{t\geq0})_{i=1,\dots,N}$ denote independent standard Brownian motions. Global information about the objective function is encoded in the consensus point $c(x)$, which is computed as

$$
c_\alpha(x) = \frac{1}{\sum_{i=1}^N \omega_\alpha(x^i_t)} \sum_{i=1}^N x^i_t\ \omega_\alpha(x^i_t), \quad\text{ with }\quad \omega_\alpha(x) = \mathrm{exp}(-\alpha f(x)).
$$

Each particle is driven by a drift towards the consensus (confinement) and a scaled diffusion (exploration). The scaling factor of the diffusion is proportional to the distance of the particle to the mean. Hence, whenever the particle's position and the location of the weighted mean coincide, the particle stops moving. Concerning the analysis of the methods, the main challange is to balance the drift and diffusion term in such a way, that all particles converge to the weighted mean which is located at the global best position of the state space. If the drift is too strong, the convergence may happen prematurely. On the other hand, if the diffusion is too strong, there may be no convergence at all. The choice of the weight function defining the mean is motived by Laplace principle [@dembo1998large], which ensured that the weighted mean converges to the position of the particle with best objective values (assuming that this particle is unique). From a computational perspective, the method is attractive as the particle interactions scale linearly with the number of particles.

A theoretical convergence analysis is not directly conducted on the above SDE system due its highly complex behavior, but on its macroscopic mean-field limit (infinite-particle limit) [@huang2021MFLCBO], which can be described by a nonlinear nonlocal Fokker-Planck equation [@pinnau2017consensus;@carrillo2018analytical;@carrillo2021consensus;@fornasier2021consensus;@fornasier2021convergence]. The implemented CBO code originates from a simple Euler-Maruyama time discretization of the above SDE system. A convergence statement therefore is available in [@fornasier2021consensus].
Similar analysis techniques further allowed to obtain theoretical convergence guarantees for PSO [@qiu2022PSOconvergence].

As of now, CBX methods have been deployed in a variety of different settings and for different purposes, such as for solving constrained optimizations [@fornasier2020consensus_sphere_convergence;@borghi2021constrained], multi-objective optimizations [@borghi2022adaptive;@klamroth2022consensus], saddle point problems [@huang2022consensus], federated learning tasks [@carrillo2023fedcbo], adversarial training [] or for sampling [@carrillo2022consensus].
In addition, recent work [@riedl2023gradient] establishes a connection of CBO to stochastic gradient descent-type methods, suggesting a more fundamental connection of theoretical interest between derivative-free and gradient-based methods.

# Features of CBXPy
![Logo of CBXPy](CBXPy.png){ width=50% }

CBXPy was designed to express common abstractions between different variants of CBO, while still allowing the freedom of customization. The term **dynamic** is used to describe an instance of certain CBO variant. Each dynamic inherits from the base class ```CBXDynamic```, which implements the base functionality, such as

* a ```step``` method, which can then be exectued in the ```solve``` method which then performs the optimization of a given objective funtion,
* an index-selection scheme, which allows to easily implement a mini-batched variant of custom CBO methods,
* different termination criteria, which can be used to stop the optimization process,
* different noise methods, such as isotropic, anisotropic or the sampling noise from [@carrillo2022consensus].

Most of the code uses basic Python functionality, where the ensemble $x$ is modeled as an array-like structure. For certain specific features, like broadcasting-beahviour, array copying and index selection we fall back to the  ```numpy``` implementation [@harris2020array], however, it should be noted that an adaption to PyTorch is straightforward. For the computation of the consenus point, we rely on the ```logsumexp``` function from ```scipy``` [@2020SciPy-NMeth], which allows for a numerical stable and efficiant evaluation of the weighted mean. Furthermore, we employ the ```matplotlib``` library [@Hunter_2007] for visualization purposes.


A simple approach for achieving parallelization, is done by running multiple instances of a single dynamic in parallel. Additionally, in the python version one has the option of low-level parallelization exploiting array operations in 
```numpy```. An ensemble has the dimension $M\times N\times d$, where $M$ is the number of runs, $N$ is the number of particles and $d$ is the dimension of the state space.

The package is available on [GitHub](https://github.com/pdips/CBXpy) and can be installed via ```pip```, since it is released on PyPI. It is licensed under the MIT license. A documentation is available [online](https://pdips.github.io/CBXpy/).

# Features of CBX.jl
![Logo of CBXPy](CBXjl.png){ width=50% }


# Acknowledgements

TR acknowledges support from DESY (Hamburg, Germany), a member of the Helmholtz Association HGF.

We thank the Lorentz Center in Leiden for their kind hospitality during the workshop "Purpose-driven particle systems" in Spring 2023, where this work was initiated.

# References
