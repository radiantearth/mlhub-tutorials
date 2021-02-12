# Radiant MLHub Tutorials

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radiantearth/mlhub-tutorials/main?filepath=notebooks%2Findex.ipynb)

This repository contains introductions and Jupyter Notebook examples on how to access Radiant MLHub API.

You can start by reading the [introductory guide](RadiantMLHub-intro.pdf), or jump into using the [Jupyter Notebook examples](./notebooks/index.ipynb) 
and interact with the API. 

## Run on Binder
You can run the Jupyter Notebook examples using [Binder](https://mybinder.org/) by clicking on the 
"launch binder" badge above or [this link](https://mybinder.org/v2/gh/radiantearth/mlhub-tutorials/main?filepath=notebooks%2Findex.ipynb). 
The Binder environment will automatically install any dependencies required by the notebooks. 

*Note that the examples involving downloading 
assets will download those assets to the remote Binder environment and not to your local file system. To download these assets locally you 
must run the notebooks locally (see next section).* 

## Run Locally

To run the notebooks locally:

1) Create & activate a virtual environment of your choice 

2) Install dependencies:

    ```shell
    pip install -r requirements_dev.txt
    ```

3) Run Jupyter Notebook server:

    ```shell
    # Example notebooks are in the notebooks/ directory
    jupyter notebook notebooks/
    ```

## Documentation
You can access the full documentation of Radiant MLHub API [here](http://docs.mlhub.earth). 

## Contribute
If you find these guides useful and would like to contribute, make a pull request or send us an email at ml@radiant.earth.
