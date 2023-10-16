# rihdata

Copyright &copy; 2023 Darren Erik Vengroff

[![Hippocratic License HL3-CL-ECO-EXTR-FFD-LAW-MIL-SV](https://img.shields.io/static/v1?label=Hippocratic%20License&message=HL3-CL-ECO-EXTR-FFD-LAW-MIL-SV&labelColor=5e2751&color=bc8c3d)](https://firstdonoharm.dev/version/3/0/cl-eco-extr-ffd-law-mil-sv.html)

This project downloads race, income, and housing data
from the U.S. Census Bureau and saves it locally in
a format suitable for modeling. It relies on the `censusdis`
package to do this. 

The primary consumer of this data is the `rihcharts` project.
That project generates impact charts using the interpreted 
ML approach to modeling relationships of variables.

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
3. Generate exploratory plots of median home price 
   vs. houshold income at the block level.

For more details and options, please consult the 
`Makefile`. The most common option to change, which
can be done via the command line, is to set the `N`
value for which the top `N` CBSAs data will be downloaded.
The default is 50, but if you only want data for the 
top 10, you can run

```shell
gmake N=10
```

