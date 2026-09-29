# LAGO

This repository contains a Python implementation of the algorithm presented in
**LAGO: A Local–Global Optimization Framework Combining Trust Region Methods and Bayesian Optimization** ([arXiv](https://arxiv.org/abs/2603.02970)).

LAGO is a hybrid optimization algorithm for expensive, smooth objectives. It combines **global exploration** via Bayesian optimization with **fast local refinement** via an SR1 trust-region method, using an adaptive selection rule to choose which candidate to evaluate at each iteration.

This is a research implementation of LAGO. The code is under active development and the API may change. **LAGO-BO is currently implemented**; numerical parity and benchmark reproduction against the original Julia implementation are ongoing.

The original Julia implementation and experimental setup are available on [Zenodo](https://zenodo.org/records/20557676).