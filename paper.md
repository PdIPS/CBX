---
title: 'CBX: Python and Julia packages of consensus-based interacting particle methods'
tags:
  - Python
  - Sampling
  - Optimization
authors:
  - name: Tim Roith
    orcid: 0000-0001-8440-2928
    affiliation: 1
affiliations:
 - name: Friedrich-Alexander-Universität Erlangen-Nürnberg
   index: 1
date: 23 August 2023
bibliography: paper.bib
---

# Summary

We present CBXpy and CBX.jl which provide Python and respectively Julia implementations for consensus-based interacting particle methods. In detail, the packages focus on consensus-based optimization (CBO) [@pinnau2017consensus] and consensus-based sampling (CBS) [@carrillo2022consensus], which coined the acronym CBX. The Python and Julia implementations were developed in parallel, in order to provide a framework for researchers more familiar with either language. Here, we focused on having a similar API and core functionality in both packages, while taking advantage of the strengths of each language, and writing idiomatic code.

![Visualization of a CBO run for the Ackley function [@ackley2012connectionist].](JOSS.png)

# Statement of need

Consensus-based optimization (CBO) was proposed in [@pinnau2017consensus] as a zeroth-order particle-based scheme, to solve problems of the form

$$
min_{x\in\mathcal{X}} f(x)
$$

for some input space $\mathcal{X}$ and a possibly non-convex objective function $f:\mathcal{X}\to\mathbb{R}$. As an agent-based method, CBO is conceptually comparable to biologically and physically inspired methods [@mohan2012survey;karaboga2014comprehensive;yang2009firefly;bayraktar2013wind], particle-swarm optimization (PSO) [@kennedy1995particle] or simulated annealing [@henderson2003theory]. However, compared to other heuristics, one can derive a limiting PDE in the infinite-particle limit, which has sparked considerable theoretical interest in recent years [@totzeck2021trends]. From a computational side, the method is also attractive, since the amount of particle interaction scales linearly with the number of particles. 

For PSO and SA there are already available Python implementations [@miranda2018pyswarms;@scikitopt]



# Acknowledgements

# References
