# Radiant MLHub Python Client Examples

The notebooks in this repository provide examples and tutorials to cover some common uses of [Radiant MLHub data](https://mlhub.earth/). All examples utilize Python in Jupyter Notebooks.

### radiant-mlhub Documentation
Each tutorial in this directory uses the radiant-mlhub package. To see the full documentation of the Radiant MLHub Python Client, visit the docs [here](https://radiant-mlhub.readthedocs.io/en/latest/). 

## Guide to Accessing Data from Radiant MLHub
Step-by-step guides for accessing datasets via Radiant MLHub. While this does not cover all datasets available in the Radiant MLHub, the principles here can be applied to accessing other datasets.

* [**CV4A Kenya Crop Type Competition Dataset**](./Guide%20to%20Accessing%20Data%20from%20Radiant%20MLHub/CV4A%20Kenya%20Crop%20Type%20Competition%20Dataset/): How use Python to see what data are available for download from Radiant MLHub, download a dataset, explore downloaded dataset, and visualize a portion of the downloaded data.  
* [**NASA Flood Extent Detection Dataset**](./Guide%20to%20Accessing%20Data%20from%20Radiant%20MLHub/NASA%20Flood%20Extent%20Detection%20Dataset/): How use Python to see what data are available for download from Radiant MLHub, create multiple filters&ndash;spatial, temporal, and for STAC collection(s)&ndash;and download a subset of a dataset using the defined filters.

## Guide to Building a Basic Machine Learning Model
Radiant MLHub provides the building blocks for model development: source imagery and labels. The tutorial in this directory demonstrate how to create a basic machine learning model using data from Radiant MLHub. 

* [**Agrifieldnet Compeition Basic Model**](./Guide%20to%20Building%20a%20Basic%20Machine%20Learning%20Model/A%20Baseline%20Model%20for%20the%20AgrifieldNet%20India%20Competition/): Load the data and build a baseline model using Random Forests on the  [ref_agrifieldnet_competition_v1](https://mlhub.earth/data/ref_agrifieldnet_competition_v1) dataset. 

* [**Basic Cloud Detection Model**](./Guide%20to%20Building%20a%20Basic%20Machine%20Learning%20Model/Basic%20Cloud%20Detection%20Model/): Explore datasets available on Radiant MLHub, download cloud cover dataset from Radiant MLHub, Populate a PyTorch DataLoader, visualize the cloud classification balance in the training dataset, define model architechture, train the model, plot the model performance, and produce a confusion matrix. 

* [**NASA Harvest Rwanda Baseline Model**](./Guide%20to%20Building%20a%20Basic%20Machine%20Learning%20Model/2021%20NASA%20Harvest%20Rwanda%20Baseline%20Model/): Create a baseline field delineation model for detecting boundaries from Sentinel-2 time-series satellite imagery using a spatio-temporal U-Net model on the [2021 NASA Harvest Rwanda dataset](https://mlhub.earth/data/nasa_rwanda_field_boundary_competition).

# How to Run the Notebooks

Each subdirectory contains its own `requirements.txt` file that contains all the dependencies needed to be able to run the subdirectory's notebook(s). To run a given set of notebooks locally:

1) Create & activate a virtual environment of your choice.

2) Change to the subdirectory you wish to run:

    ```shell
    cd tutorials/<TARGET DIRECTORY>
    ```

3) Install dependencies:

    ```shell
    pip install -r requirements.txt
    ```

4) Run Jupyter Notebook server:

    ```shell
    jupyter notebook
    ``` 