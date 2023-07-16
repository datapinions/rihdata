# rihmodel

Copyright &copy; 2023 Darren Erik Vengroff

[![Hippocratic License HL3-CL-ECO-EXTR-FFD-LAW-MIL-SV](https://img.shields.io/static/v1?label=Hippocratic%20License&message=HL3-CL-ECO-EXTR-FFD-LAW-MIL-SV&labelColor=5e2751&color=bc8c3d)](https://firstdonoharm.dev/version/3/0/cl-eco-extr-ffd-law-mil-sv.html)

A demonstration of the interpreted ML approach to modeling relationships of variables.

**This project is in it's early stages. Check back soon for more.**

## Instructions

All of the data loading and analysis this project does is
orchestrated by a single `Makefile`. It requires
[GNU make](https://www.gnu.org/software/make/) 
version 4.3 or higher.

Before you run GNU make, you will have to set up a
Python virtual environment with all the project's
dependencies. I use [poetry](https://python-poetry.org/) 
to do this. All top-level dependencies are listed in 
`pyproject.toml` and the full set of recursive 
dependencies and version is stored in `poetry.lock`.

Once you have your virtual environment and the right
version of GNU Make installed, the single command

```shell
gmake
```

should build the entire project. The steps it will go
through, driven by the Makefile, are:

1. Download data from the top 50 largest CBSAs.
2. Generate features for each CBSA.
3. Generate exploratory plots of median home price vs. houshold income at the block level.
4. Fit and hyperparameter tune an XGBoost model 
   for each CBSA that predicts median housing
   prices based on median income and racial demographics.
5. Fit a series of linear models for the same problem.
6. Generate SHAP values and plots for a large number 
   of individual models and their ensemble for each CBSA and demographic feature.

How long this will take depends on how fast your 
machine and your internet connection are. For me,
from a clean state it takes about an hour to complete.
