# Radiant MLHub API Examples

The notebooks below provide examples and tutorials to cover some common uses of Radiant MLHub data. All examples utilize Python in Jupyter Notebooks.

## radiant-mlhub Documentation
Each tutorial in this repository uses the radiant-mlhub package. To see the full documentation of the Radiant MLHub Python Client, visit the docs [here](https://mlhub.earth/docs). 

## Guide to Accessing Data from Radiant MLHub
Step-by-step guides for accessing datasets via Radiant MLHub. While this does not cover all datasets available in the API, the principles here can be applied to accessing other datasets.

* [**CV4A Kenya Crop Type Competition Dataset**](./Guide to Accessing Data from Radiant MLHub/CV4A Kenya Crop Type Competition Dataset/)
* [**NASA Flood Extent Detection Dataset**](./Guide to Accessing Data from Radiant MLHub/NASA Flood Extent Detection Dataset/)


## Guide to Building a Basic Machine Learning Model
Radiant MLHub provides the building blocks for model development: source imagery and labels. The tutorials in this subdirectory demonstrate how to create a basic ML model using data from Radiant MLHub. 
Cloud Detection Model Development

# Run in Hosted Environment

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

### Run Locally

To run the notebooks locally:

1) Create & activate a virtual environment of your choice 

2) Change to the subdirectory you wish to run.

3) Install dependencies:

    ```shell
    pip install -r requirements.txt
    ```

3) Run Jupyter Notebook server:

    ```shell
    # Check to make sure you are in the directory where the notebook(s) you want to run live
    jupyter notebook
    ```