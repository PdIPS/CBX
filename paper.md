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
    ordic: 0000-0001-6283-7154
    affiliation: 4  
affiliations:
 - name: Friedrich-Alexander-Universität Erlangen-Nürnberg
   index: 1
 - name: Technical University of Munich
   index: 2
 - name: Munich Center for Machine Learning
   index: 3
 - name: University of Wuppertal
   index: 4 
date: 09 October 2023
bibliography: paper.bib
---

# Summary

We present CBXpy and CBX.jl which provide Python and Julia implementations, respectively, for consensus-based interacting particle methods. In detail, the packages focus on consensus-based optimization (CBO) [@pinnau2017consensus] and consensus-based sampling (CBS) [@carrillo2022consensus], which coined the acronym CBX. The Python and Julia implementations were developed in parallel, in order to provide a framework for researchers more familiar with either language. While we focused on having a similar API and core functionality in both packages, we took advantage of the strengths of each language and wrote idiomatic code.

![Visualization of a CBO run for the Ackley function [@ackley2012connectionist].](JOSS.png){ width=50% }

# Statement of need

Consensus-based optimization (CBO) was proposed in [@pinnau2017consensus] as a zero-order (derivative-free) particle-based scheme, to solve problems of the form
$$
x^* = \mathrm{argmin}_{x\in\mathcal{X}} f(x),
$$
for some input space $\mathcal{X}$ and a possibly nonconvex and nonsmooth objective function $f:\mathcal{X}\to\mathbb{R}$. As a particle method, CBO is conceptually comparable to biologically and physically inspired methods such as particle-swarm optimization (PSO) [@kennedy1995particle], simulated annealing (SA) [@henderson2003theory] or several other heuristics [@mohan2012survey;@karaboga2014comprehensive;@yang2009firefly;@bayraktar2013wind]. However, compared to these methods, CBO was designed to be amenable to a rigorous theoretical convergence analysis on the mean-field level [@carrillo2018analytical;@carrillo2021consensus;@fornasier2021consensus;@fornasier2021convergence]. To this end, global information about the objective funtion is encoded in a weighted average making the particles indistinguishable. Each particle is driven by a drift towards this weighted mean (confinement) and a scaled diffusion (exploration). The scaling factor of the diffusion is proportional to the distance of the particle to the mean. Hence, whenever the particle's position and the location of the weighted mean coincide, the particle stops moving. Concerning the analysis of the methods, the main challange is to balance the drift and diffusion term in such a way, that all particles converge to the weighted mean which is located at the global best position of the state space. If the drift is too strong, the convergence may happen prematurely. On the other hand, if the diffusion is too strong, there may be no convergence at all. The choice of the weight function defining the mean is motived by Laplace principle [@dembo1998large], which ensured that the weighted mean converges to the position of the particle with best objective values (assuming that this particle is unique). From a computational perspective, the method is attractive as the particle interactions scale linearly with the number of particles. 

For Python, PSO and SA implementations are already available [@miranda2018pyswarms;@scikitopt;@deapJMLR2012;@pagmo2017], which are widely used in the community and provide a rich framework for the respective methods. However, adjusting these implementations to CBO is not straightforward. Furthermore, in this project, we want to provide a lightweight and direct implementation of CBO methods, which are easy to understand and to modify. The first publicly available Python packages implementing CBO-type algorithms were given by some of the authors together with collaborators in [@Igor_CBOinPython], where CBO as in [@pinnau2017consensus] is implemented, as well as in [@Roith_polarcbo], where so-called polarized CBO [@bungert2022polarized] is implemented. The current Python package is a complete rewrite of the latter implementation.

For Julia, PSO and SA methods are, among others, implemented in [@mogensen2018optim;@mejia2022metaheuristics;@Bergmann2022]. Similarly, one of the authors provided the first specific Julia implementation of CBO [@Bailo_consensus]. However, the current version of the package deviates from the previous implementation and is more closely oriented toward the Python implementation.

We summarize the motivation and main features of the packages in what follows.

- Provide a lightweight, easy-to-understand, -use and -extend implementation of CBO together with several of its variants. These include CBO with mini-batching [@carrillo2021consensus], polarized CBO [@bungert2022polarized], CBO with memory effects [@grassi2020particle;@riedl2022leveraging], and CBS [@carrillo2022consensus]. The implementation relies on ...
- torch and tensorflow-like usage style (and implementation way via step), maybe provide here code snippet
- 

# Mathematical background

CBO methods use a finite number of agents $X^1,\dots,X^N$ to explore the domain and to form a global consensus about the location of the minimizer $x^*$ as time passes. They are described through a system of stochastic differential equations (SDEs), expressed in It\^o's form as
$$
dX^i_t = -\lambda (X^i_t-x_\alpha(\widehat\rho_t^N)) dt + \sigma D(X^i_t-x_\alpha(\widehat\rho_t^N)) dB^i_t,
$$
where $\alpha,\lambda$ and $\sigma$ are user-specified parameters, $((B^i_t)_{t\geq0})_{i=1,\dots,N}$ denote independent standard Brownian motions and where $x_\alpha(\widehat\rho_t^N)$ denotes the consensus point, a suitably weighted average of the positions of the particles, which is computed as
$$
x_\alpha(\widehat\rho_t^N) = \frac{1}{\sum_{i=1}^N \omega_\alpha(X^i_t)} \sum_{i=1}^N X^i_t\omega_\alpha(X^i_t), \quad\text{ with }\quad \omega_\alpha(x) = \mathrm{exp}(-\alpha f(x)).
$$
A theoretical convergence analysis is not directly conducted on the above SDE system due its highly complex behavior, but on its macroscopic mean-field limit (infinite-particle limit) [@huang2021MFLCBO], which can be described by a nonlinear nonlocal Fokker-Planck equation [@pinnau2017consensus;@carrillo2018analytical;@carrillo2021consensus;@fornasier2021consensus;@fornasier2021convergence]. The implemented CBO code originates from a simple Euler-Maruyama time discretization of the above SDE system. A convergence statement therefore is available in [@fornasier2021consensus].
Similar analysis techniques further allowed to obtain theoretical convergence guarantees for PSO [@qiu2022PSOconvergence].

# Application areas of CBX

As of now, CBX methods have been deployed in a variety of different settings and for different purposes, such as for solving constrained optimizations [@fornasier2020consensus_sphere_convergence;@borghi2021constrained], multi-objective optimizations [@borghi2022adaptive;@klamroth2022consensus], saddle point problems [@huang2022consensus], federated learning tasks [@carrillo2023fedcbo], adversarial training [] or for sampling [@carrillo2022consensus].
In addition, recent work [@riedl2023gradient] establishes a connection of CBO to stochastic gradient descent-type methods, suggesting a more fundamental connection of theoretical interest between derivative-free and gradient-based methods.

# Acknowledgements

We thank the Lorentz Center in Leiden for their kind hospitality during the workshop Purpose-driven particle systems in Spring 2023, where this work was initiated.

# References
