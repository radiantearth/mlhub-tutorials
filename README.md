# Radiant MLHub Tutorials

The notebooks in this repository provide examples and tutorials to cover some common uses of [Radiant MLHub data](https://mlhub.earth/). All examples utilize Python in Jupyter Notebooks.

To view an overview of the lessons available in this repository, visit this [tutorial overview](./notebooks/README.md).

## How to Run the Notebooks

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

3) Run Jupyter Notebook server:

    ```shell
    jupyter notebook
    ```

## Documentation
You can access the full documentation of Radiant MLHub API and Python Client [here](https://mlhub.earth/docs). 

## Contribute
If you find these guides useful and would like to contribute, make a pull request or send us an email at ml@radiant.earth.
