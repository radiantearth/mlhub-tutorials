This tutorial is divided into four notebooks. Each notebook performs a specific task with the goal of developing a machine learning model on the South Africa field boundary detection dataset as follows:

1. [Extract_data.ipynb](https://github.com/radiantearth/mlhub-tutorials/blob/mali-crop-type/notebooks/South-Africa-Field-Boundary/1.%20Extract_data.ipynb): This notebooks focuses on extracting the data from an S3 bucket and combining the RGB bands for further processing.
After running this notebook, you should have a file structure as follows:
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

All you need to run is 
`pip install -r requirements.txt`
Or `Conda install --file requirements.txt`

Note that in using Conda install, you might get NoPackagesFound for the following libraries:

  - torch
  - opencv-python
  - keras==2.3.1
  - segmentation_models

All you need to do is to run a pip install for these packages
