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

* [**Basic Model Training**](./Guide%20to%20Building%20a%20Basic%20Machine%20Learning%20Model/Basic%20Model%20Training/): Explore datasets available on Radiant MLHub, download cloud cover dataset from Radiant MLHub, Populate a PyTorch DataLoader, visualize the cloud classification balance in the training dataset, define model architechture, train the model, plot the model performance, and produce a confusion matrix. 



# How to Run the Notebooks

Each subdirectory contains its own `requirements.txt` file that contains all the dependencies needed to be able to run the subdirectory's notebook(s). To run a given set of notebooks locally:

1) Create & activate a virtual environment of your choice.


2) Change to the subdirectory you wish to run:
    ```shell
    cd notebooks/<TARGET DIRECTORY>
    ```

3) Install dependencies:

    ```shell
    pip install -r requirements.txt
    ```

3) Run Jupyter Notebook server:

    ```shell
    jupyter notebook
    ```