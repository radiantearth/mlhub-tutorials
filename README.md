# Radiant MLHub Tutorials

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/radiantearth/mlhub-tutorials/main?filepath=notebooks%2Findex.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/radiantearth/mlhub-tutorials/blob/main/notebooks/index.ipynb)

This repository contains introductions and Jupyter Notebook examples on how to access Radiant MLHub API.

You can start by reading the [introductory guide](RadiantMLHub-intro.md), or jump into using the [Jupyter Notebook examples](./notebooks/index.ipynb) 
and interact with the API. 

## Run in Hosted Environment

*Note that the examples involving downloading assets will download those assets to the remote environment and not to 
your local file system. To download these assets locally you must run the notebooks locally (see next section).*

### Run on Binder
You can run the Jupyter Notebook examples using [Binder](https://mybinder.org/) by clicking on the 
"launch binder" badge above or [this link](https://mybinder.org/v2/gh/radiantearth/mlhub-tutorials/main?filepath=notebooks%2Findex.ipynb). 
The Binder environment will automatically install any dependencies required by the notebooks. 

### Run in Google Colab
You can run the Jupyter Notebook examples using [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#) 
by clicking on the "Open in Colab" badge above or [this 
link](https://colab.research.google.com/github/radiantearth/mlhub-tutorials/blob/main/notebooks/index.ipynb). Colab does 
not have a mechanism for automatically installing dependencies to a notebook environment like Binder, so *you will need to 
install all dependencies within the notebook as follows:*

```
%pip install radiant-mlhub~=0.5.1 tifffile==2019.7.26.2 pandas~=1.2.0 matplotlib~=3.3.4 scikit-image~=0.18.1
```

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
You can access the full documentation of Radiant MLHub API [here](https://mlhub.earth/docs). 

## Contribute
If you find these guides useful and would like to contribute, make a pull request or send us an email at ml@radiant.earth.
