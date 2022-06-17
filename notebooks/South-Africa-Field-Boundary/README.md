This tutorial is divided into four notebooks. Each notebook performs a specific task with the goal of developing a machine learning model on the South Africa field boundary detection dataset as follows:

1. [Extract_data.ipynb](https://github.com/radiantearth/mlhub-tutorials/blob/mali-crop-type/notebooks/South-Africa-Field-Boundary/1.%20Extract_data.ipynb): This notebooks focuses on extracting the data from an S3 bucket and combining the RGB bands for further processing.
After running this notebook, you should have a base folder structure as follows:
```
Working Directory    
│
└───data
│   └───extracted_data
│         └───imagery
│         └───labels
│   └───images
```

2. [Data Augmentation.ipynb](https://github.com/radiantearth/mlhub-tutorials/blob/mali-crop-type/notebooks/South-Africa-Field-Boundary/2.%20Data%20Augmentation.ipynb): This notebook focuses on augmenting the extracted RGB data. These augmentation techniques were thanks to the radix-ai GitHub repository, which can be accessed [here](https://github.com/radix-ai/agoro-field-boundary-detector). 
For the purpose of data augmentation, only the [`src`](https://github.com/radix-ai/agoro-field-boundary-detector/tree/master/src/agoro_field_boundary_detector) folder from the repo was used.

3. [Data Preparation.ipynb](https://github.com/radiantearth/mlhub-tutorials/blob/mali-crop-type/notebooks/South-Africa-Field-Boundary/3.%20Data%20Preparation.ipynb): Here, we will prepare the augmented training data into training, validation and test data. We will also check for augmented images with empty labels/masks and exclude them from model training.

4. [Model Training.ipynb](https://github.com/radiantearth/mlhub-tutorials/blob/mali-crop-type/notebooks/South-Africa-Field-Boundary/4.%20Model%20Training.ipynb). This notebook delves into building the UNet-Agri model and training it on the training data. 

To excute all notebooks, you need to install the needed libraries from the [`requirements.txt`](https://github.com/radiantearth/mlhub-tutorials/blob/mali-crop-type/notebooks/South-Africa-Field-Boundary/requirements.txt) file.

All you need to do is run on your terminal:
`pip install -r requirements.txt`
or `conda install --file requirements.txt`. The latter is very advisable if you're using Apple Silicon (M1 or similar). Also note that you'd need to install tensorflow-macos rather than the usual tensorflow if using the M1 or similar.

Note that in using conda install, you might get NoPackagesFound for the following libraries:

  - torch
  - opencv-python
  - keras==2.3.1
  - segmentation_models

All you need to do is to run a pip install for these packages individually.

To install rasterio, you might also need to download both binaries for rasterio and GDAL. You can read the [documentation](https://rasterio.readthedocs.io/en/latest/installation.html) here  
