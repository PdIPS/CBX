---
title: 'CBX: Python and Julia Packages for Consensus-Based Interacting Particle Methods'
tags:
  - Python
  - Julia
  - Optimisation
  - Sampling
authors:
  - name: Tim Roith
    orcid: 0000-0001-8440-2928
    affiliation: 1
  - name: Konstantin Riedl
    orcid: 0000-0002-2206-4334
    affiliation: '2, 3'
  - name: Claudia Totzeck
    orcid: 0000-0001-6283-7154
    affiliation: 4
  - name: Alethea Barbaro
    orcid: 0000-0001-9856-2818
    affiliation: 5
  - name: Susana N. Gomes
    orcid: 0000-0002-8731-367X
    affiliation: 6
  - name: Urbain Vaes
    orcid: 0000-0002-7629-7184
    affiliation: '7, 8'
  - name: Rafael Bailo
    orcid: 0000-0001-8018-3799
    affiliation: 9
affiliations:
  - name: Helmholtz Imaging, Deutsches Elektronen-Synchrotron DESY, Notkestr. 85, 22607 Hamburg, Germany
    index: 1
  - name: Technical University of Munich
    index: 2
  - name: Munich Center for Machine Learning
    index: 3
  - name: University of Wuppertal
    index: 4
  - name: Delft University of Technology
    index: 5
  - name: Mathematics Institute, University of Warwick
    index: 6
  - name: MATHERIALS team, Inria Paris
    index: 7
  - name: École des Ponts
    index: 8
  - name: Mathematical Institute, University of Oxford
    index: 9
date: 15 March, 2024
bibliography: paper.bib
---

# Summary

We introduce [CBXPy](https://pdips.github.io/CBXpy/) and [ConsensusBasedX.jl](https://pdips.github.io/ConsensusBasedX.jl/), Python and Julia implementations of consensus-based interacting particle systems (CBX), which generalise consensus-based optimization methods (CBO) for global, derivative-free optimisation.  The _raison d'être_ of our libraries is twofold: on the one hand, to offer high-performance implementations of CBX methods that the community can use directly without the need to program from scratch, while on the other, providing a general interface that can accommodate and be extended to further variations of the CBX family. Python and Julia were selected as the leading high-level languages in terms of usage and performance, as well as their popularity among the scientific computing community. Both libraries have been developed with a common _ethos_, ensuring a similar API and core functionality, while leveraging the strengths of each language and writing idiomatic code.

# Mathematical background

Consensus-based optimisation (CBO) is an approach to solve, for a given (continuous) _objective function_ $f:\mathbb{R}^d \rightarrow \mathbb{R}$, the _global minimisation problem_

$$
x^* = \operatorname*{argmin}_{x\in\mathbb{R}^d} f(x),
$$

i.e., the task of finding the point $x^*$ where $f$ attains its lowest value. Such problems arise in a variety of disciplines including engineering, where $x$ might represent a vector of design parameters for a structure and $f$ a function related to its cost and structural integrity, or machine learning, where $x$ could comprise the parameters of a neural network and $f$ the empirical loss, which measures the discrepancy of the neural network prediction with the observed data.

In some cases, so-called _gradient-based methods_ (those that involve updating a guess of $x^*$ by evaluating the gradient $\nabla f$) achieve state-of-the-art performance in the global minimisation problem. However, in scenarios where $f$ is _non-convex_ (when $f$ could have many _local minima_), where $\nabla f$ is not well-defined, or where the evaluation of $\nabla f$ is impractical due to cost or complexity, _derivative-free_ methods are needed. Numerous techniques exist for derivative-free optimisation, such as _random_ or _pattern search_ [@friedman1947planning;@rastrigin1963convergence;@hooke1961direct], _Bayesian optimisation_ [@movckus1975bayesian] or _simulated annealing_ [@henderson2003theory]. Here, we focus on _particle-based methods_, specifically, consensus-based optimisation (CBO), as proposed by @pinnau2017consensus, and the consensus-based taxonomy of related techniques, which we term _CBX_.

CBO uses a finite number $N$ of _agents_ (particles), $x_t=(x_t^1,\dots,x_t^N)$, to explore the landscape of $f$ without evaluating any of its derivatives (as do other CBX methods). At each time $t$, the agents evaluate the objective function at their position, $f(x_t^i)$, and define a _consensus point_ $c_\alpha$. This point is an approximation of the global minimiser $x^*$, and is constructed by weighing each agent's position against the _Gibbs-like distribution_ $\exp(-\alpha f)$ [@boltzmann1868studien]. More rigorously,

$$
c_\alpha(x_t) =
\frac{1}{ \sum_{i=1}^N \omega_\alpha(x_t^i) }
\sum_{i=1}^N x_t^i \, \omega_\alpha(x_t^i),
\quad\text{where}\quad
\omega_\alpha(\,\cdot\,) = \mathrm{exp}(-\alpha f(\,\cdot\,)),
$$

for some $\alpha>0$. The exponential weights in the definition favour those points $x_t^i$ where $f(x_t^i)$ is lowest, and comparatively ignore the rest, particularly for larger $\alpha$. If all the found values of the objective function are approximately the same, $c_\alpha(x_t)$ is roughly an arithmetic mean. Instead, if one particle is much better than the rest, $c_\alpha(x_t)$ will be very close to its position.

Once the consensus point is computed, the particles evolve in time following the _stochastic differential equation_ (SDE)

$$
\mathrm{d}x_t^i =
-\lambda\ \underbrace{
\left( x_t^i - c_\alpha(x_t) \right) \mathrm{d}t
}_{
\text{consensus drift}
}
+ \sigma\ \underbrace{
\left\| x_t^i - c_\alpha(x_t) \right\| \mathrm{d}B_t^i
}_{
\text{scaled diffusion}
},
$$

where $\lambda$ and $\sigma$ are positive parameters, and where $B_t^i$ are independent Brownian motions in $d$ dimensions. The _consensus drift_ is a deterministic term that drives each agent towards the consensus point, with rate $\lambda$. Meanwhile, the _scaled diffusion_ is a stochastic term that encourages exploration of the landscape. While both the agents' positions and the consensus point evolve in time, one could reasonably expect that all agents eventually reach the same position and that the consensus point $c_\alpha(x_t)$ is a good approximation of $x^*$. Other variations of the method, such as CBO with anisotropic noise [@carrillo2021consensus], _polarised CBO_ [@bungert2022polarized], or _consensus-based sampling_ (CBS) [@carrillo2022consensus] have also been proposed.

In practice, the solution to the SDE above cannot be found exactly. Instead, an _Euler-Maruyama scheme_ [@KP1992] is used to update the position of the agents. The update is given by

$$
x^i \gets x^i
-\lambda \,\Delta t
\left( x^i - c_\alpha(x) \right)
+ \sigma\sqrt{\Delta t}\
\left\| x^i - c_\alpha(x) \right\| \xi^i,
$$

where $\Delta t > 0$ is the _step size_ and $\xi^i \sim \mathcal{N}(0,\mathrm{Id})$ are independent, identically distributed, standard normal random vectors.

As a particle-based family of methods, CBX is conceptually related to other optimisation approaches which take inspiration from biology, like _particle-swarm optimisation_ (PSO) [@kennedy1995particle], from physics, like _simulated annealing_ (SA) [@henderson2003theory], or from other heuristics [@mohan2012survey;@karaboga2014comprehensive;@yang2009firefly;@bayraktar2013wind]. However, unlike many such methods, CBX has been designed to be compatible with rigorous convergence analysis at the mean-field level (the infinite-particle limit, see [@huang2021MFLCBO]). Many convergence results have been shown, whether in the original formulation [@carrillo2018analytical;@fornasier2021consensus], for CBO with anisotropic noise [@carrillo2021consensus;@fornasier2021convergence], with memory effects [@riedl2022leveraging], with truncated noise [@fornasier2023consensus], for polarised CBO [@bungert2022polarized], and PSO [@qiu2022PSOconvergence]. The relation between CBO and _stochastic gradient descent_ has been recently established by @riedl2023gradient, which suggests a previously unknown yet fundamental connection between derivative-free and gradient-based approaches.

![Typical evolution of a CBO method minimising the Ackley function [@ackley2012connectionist].](JOSS.png){ width=100% }

CBX methods have been successfully applied and extended to several different settings, such as constrained optimisation problems [@fornasier2020consensus_sphere_convergence;@borghi2021constrained], multi-objective optimisation [@borghi2022adaptive;@klamroth2022consensus], saddle-point problems [@huang2022consensus], federated learning tasks [@carrillo2023fedcbo], uncertainty quantification [@althaus2023consensus], or sampling [@carrillo2022consensus].

# Statement of need

In general, very few implementations of CBO already exist, and none have been designed with the generality of other CBX methods in mind. We summarise here the related software:

Regarding Python, we refer to @duan2023pypop7 and @scikitopt for a collection of various derivative-free optimisation strategies. A very recent implementation of Bayesian optimisation is described by @Kim2023. PSO and SA implementations are already available [@miranda2018pyswarms;@scikitopt;@deapJMLR2012;@pagmo2017]. They are widely used by the community and provide a rich framework for the respective methods. However, adjusting these implementations to CBO is not straightforward. The first publicly available Python packages implementing CBX algorithms were given by some of the authors together with collaborators. @Igor_CBOinPython implement standard CBO [@pinnau2017consensus], and @Roith_polarcbo provide an implementation of polarised CBO [@bungert2022polarized]. [CBXPy](https://pdips.github.io/CBXpy/) is a significant extension of the latter.

Regarding Julia, PSO and SA methods are, among others, implemented by @mogensen2018optim, @mejia2022metaheuristics, and @Bergmann2022. PSO and SA are also included in the meta-library [@DR2023], as well as Nelder-Mead, which is a direct search method. One of the authors gave the first specific Julia implementation of standard CBO [@Bailo_consensus]; that package has now been deprecated in favour of [ConsensusBasedX.jl](https://pdips.github.io/ConsensusBasedX.jl/), which offers additional CBX methods and a far more general interface.

# Features

[CBXPy](https://pdips.github.io/CBXpy/) and [ConsensusBasedX.jl](https://pdips.github.io/ConsensusBasedX.jl/) provide a lightweigh and easy-to-understand high-level interface. An existing function can be optimised with just one call. Method selection, parameters, different approaches to particle initialisation, and termination criteria can be specified directly through this interface, offering a flexible point of entry for the casual user. Some of the methods provided are standard CBO [@pinnau2017consensus], CBO with mini-batching [@carrillo2021consensus], polarised CBO [@bungert2022polarized], CBO with memory effects [@grassi2020particle;@riedl2022leveraging], and consensus-based sampling (CBS) [@carrillo2022consensus]. Parallelisation tools are available.

A more proficient user will benefit from the fully documented interface, which allows the specification of advanced options (e.g., debug output, the noise model, or the numerical approach to the matrix square root of the covariance matrix). Both libraries offer performance evaluation methods as well as visualisation tools.

Ultimately, a low-level interface (including documentation and full-code examples) is provided. Both libraries have been designed to express common abstractions in the CBX family while allowing customisation. Users can easily implement new CBX methods or modify the behaviour of the existing implementation by strategically overriding certain hooks. The stepping of the methods can also be controlled manually.

## CBXPy specifics

![CBXPy logo.](CBXPy.png){ width=50% }

Most of the [CBXPy](https://pdips.github.io/CBXpy/) implementation uses basic Python functionality, and the agents are handled as an array-like structure. For certain specific features, like broadcasting-behavior, array copying, and index selection, we fall back to the `numpy` implementation [@harris2020array]. However, it should be noted that an adaptation to other array or tensor libraries like PyTorch [@paszke2019pytorch] is straightforward. Compatibility with the latter enables gradient-free deep learning directly on the GPU, as demonstrated in the documentation.

The library is available on [GitHub](https://github.com/pdips/CBXpy) and can be installed via `pip`. It is licensed under the MIT license. The [documentation](https://pdips.github.io/CBXpy/) is available online.

## ConsensusBasedX.jl specifics

![ConsensusBasedX.jl logo.](CBXjl.png){ width=50% }

[ConsensusBasedX.jl](https://pdips.github.io/ConsensusBasedX.jl/) has been almost entirely written in native Julia (with the exception of a single call to LAPACK). The code has been developed with performance in mind, thus the critical routines are fully type-stable and allocation-free. A specific tool is provided to benchmark a typical method iteration, which can be used to detect allocations. Through this tool, unit tests are in place to ensure zero allocations in all the provided methods. The benchmarking tool is also available to users, who can use it to test their implementations of $f$, as well as any new CBX methods.

The library is available on [GitHub](https://github.com/PdIPS/ConsensusBasedX.jl). It has been registered in the [general Julia registry](https://github.com/JuliaRegistries/General), and therefore it can be installed by running `]add ConsensusBasedX`. It is licensed under the MIT license. The [documentation](https://pdips.github.io/ConsensusBasedX.jl/) is available online.

# Acknowledgements

We thank the Lorentz Center in Leiden for their kind hospitality during the workshop "Purpose-driven particle systems" in Spring 2023, where this work was initiated. RB was supported by the Advanced Grant Nonlocal-CPD (Nonlocal PDEs for Complex Particle Dynamics: Phase Transitions, Patterns and Synchronisation) of the European Research Council Executive Agency (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No. 883363) and by the EPSRC grant EP/T022132/1 "Spectral element methods for fractional differential equations, with applications
in applied analysis and medical imaging".
KR acknowledges support from the German Federal Ministry of Education and Research and the Bavarian State Ministry for Science and the Arts.
TR acknowledges support from DESY (Hamburg, Germany), a member of the Helmholtz Association HGF. This research was supported in part through the Maxwell computational resources operated at Deutsches Elektronen-Synchrotron DESY, Hamburg, Germany.
UV acknowledges support from the Agence Nationale de la Recherche under grant ANR-23-CE40-0027 (IPSO).

# References
