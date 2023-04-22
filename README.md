# PUMS Data Analysis

_(supports survey years 2012 - 2023+)_

This set of scripts and dictonaries facilitate download and analysis of Census Bureau Public Use Microdata.

Dictonary files included a reduced human-readable set of columns.

## Execution Sequence:
- [__`python`__ `get_data.py`] Downloads data for years, states, and categories outlined in `_constants.py`.
- [__`python`__ `transform.py`] Transforms values & column names
- [__`python`__ `train_test_split.py`] (Merge up into transform?) prepares train and test data sets stratified by the class of interest.
- [__`python`__ `features.py`] Prepares correlation matrices to better understand variables.
- [__`python`__ `linear_models.py`] Executes linear models for classification of test data.
- [__`python`__ `k_nn.py`] Executes a K-nn model for classification of test data.
- [__`python`__ `bayes.py`] Executes a naive Bayesian classification of test data.
- [__`python`__ `decision_tree.py`] Executes a decision tree classification of test data.
- [__`python`__ `random_forest.py`] Executes a random foreset classification of test data.

## TODO:
- feature selection?