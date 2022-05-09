# Radiant Earth Spot the Crop Challenge Tutorials

This folder contains two notebook tutorials for the Radiant Earth Spot the Crop Challenge. The dataset for this challenge consists of time-series of satellite data from Sentinel-2 multi-spectral observations and Sentinel-1 Synthetic Aperture Radar (SAR) observations. 

You can use the [nnsnsnns](south_africa_crop_type_competition_load_asset_paths.ipynb) notebook to load the data. You have the option to download and load both Sentinel-2 and Sentinel-1, or just one of them. *Note* that for the hackathon running July 2-4, 2021 you only need Sentinel-1 data, so avoid downloading Sentinel-2 which has a large file size. 

The second notebook [Radiant_Earth_Spot_the_Crop_Baseline_Model](Radiant_Earth_Spot_the_Crop_Baseline_Model.ipynb) contains a baseline model using Random Forests to walk you through the process of loading and structuring the data, training the model, and exporting the predictions to the sample CSV file. 

 
