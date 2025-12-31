**Overview**

This repository contains the tools to generate synthetic 3D building models (Flat, Gable, and Hip) and perform a systematic Monte Carlo analysis of volumetric errors. The goal is to identify the optimal "binning" statistics (Mean, Median, Maximum, P90, P95) for urban 3D modeling.

**Key Features**

- Synthetic Scene Generation: Procedural creation of $10\text{ m} \times 15\text{ m}$ buildings with predefined mathematical volumes (Flat, Gable, Hip).
- Monte Carlo Simulation: Runs 100 iterations per configuration to ensure statistically significant results.
- Accuracy Metrics: Calculates Relative Volumetric Error (RVE) across varying point densities ($2\text{–}16\text{ pts/m}^2$) and resolutions ($0.25\text{–}2.0\text{ m}$).
- Precision & Stability Analysis: Analyzes the standard deviation of RVE across all iterations to quantify the reliability and algorithmic stability of each binning method.

**Requirements**

Python 3.8+
pandas, numpy, laspy, scipy

