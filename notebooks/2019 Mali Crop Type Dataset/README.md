# Radiant Earth Mali Crop Type Tutorials

This folder contains two notebook tutorials to create a baseline model for the Mali Crop Type dataset. The dataset for this challenge consists of time-series of satellite data from Sentinel-2 multi-spectral observations.

You can use the [1. Data Preparation](https://github.com/radiantearth/mlhub-tutorials/blob/mali-crop-type/notebooks/2019%20Mali%20Crop%20Type%20Dataset/1.%20Data%20Preparation.ipynb) notebook to load the data. 

The second notebook [2. Model training](https://github.com/radiantearth/mlhub-tutorials/blob/mali-crop-type/notebooks/2019%20Mali%20Crop%20Type%20Dataset/2.%20Model%20Training.ipynb) contains a baseline model using U-Net to walk you through the process of loading and structuring the data, training the model, and visualizing the results.


## System Requirements

* Git client
* [Docker](https://www.docker.com/) with
    [Compose](https://docs.docker.com/compose/) v1.28 or newer.

## Hardware Requirements

|Inferencing|Training|
|-----------|--------|
|  4GB RAM  | 8GB RAM|
|           | NVIDIA GPU |

## Get Started With Inferencing

First clone this Git repository.
```bash
git clone https://github.com/radiantearth/mlhub-tutorials/tree/mali-crop-type/notebooks/2019%20Mali%20Crop%20Type%20Dataset
cd 2019 Mali Crop Type Dataset/
```
TODO: Change links after merge

## Building and Running a Docker Container for the Machine Learning Model
In order to start building a Docker container for a machine learning model, let’s consider three files: [Dockerfile](https://github.com/radiantearth/mlhub-tutorials/blob/mali-crop-type/notebooks/2019%20Mali%20Crop%20Type%20Dataset/Dockerfile), [train.py](https://github.com/radiantearth/mlhub-tutorials/blob/mali-crop-type/notebooks/2019%20Mali%20Crop%20Type%20Dataset/train.py), [inference.py](https://github.com/radiantearth/mlhub-tutorials/blob/mali-crop-type/notebooks/2019%20Mali%20Crop%20Type%20Dataset/inference.py).

The train.py is a python script that ingests the Mali crop type data and trains the U-Net model to perform crop type segmentation.


The inference.py will be called to perform inference by loading the model that has been previously created. The application will perform inference on test dataset and print the accuracy and loss.


To build the docker image, we run the following command in our terminal:
`docker build -t docker-ml-model -f Dockerfile .`


To perform inference on the test data, we run the following command:
`docker run docker-ml-model python3 inference.py`
