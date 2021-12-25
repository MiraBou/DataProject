# Python Data Project

## Requirements
- Python 3.7 or higher.
#### - Install pipenv on your global python setup
```Python
    pip install pipenv 
```
Or follow [documentation](https://pipenv.pypa.io/en/latest/install/) to install it properly on your system
#### - Install requirements
```sh
    cd DataProject
```
```Python
    pipenv install
```
```Python
    pipenv shell
```
#### - Start the application
```sh
    sh run.sh
```
- API : http://localhost:5000
- Streamlit Dashboard : http://localhost:8000

## Description
This mini project is a data app that revolves around credit card fraud detection.

You are given a `dataset` that contains a number of transactions.

Each row of the dataset contains:
- Features that were extracted using dimensionality reduction with `PCA` 
- The transaction amount
- A flag `[0,1]` that tells you whether a transaction is clear or fraudulent.

The project contains by default:
- A baseline `decision tree model` trained on the aforementioned dataset
- An `API` that exposes an `inference endpoint` for predictions using the baseline model
- A streamlit dashboard divided on three parts `(Exploratory Data Analysis, Training, Inference)`

Assignement given by https://gitlab.com/octomaroc_stage2022/python-data-assignment.git
