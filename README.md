# PUMS Data Analysis

_(supports survey years 2012 - 2019)_

This set of scripts and dictonaries facilitate download and analysis of Census Bureau Public Use Microdata.

Dictonary files included a reduced human-readable set of columns.

## Execution Sequence:
### 1) Data Preparation
- [__`python`__ `get_data.py`] Downloads data for years, states, and categories outlined in `_constants.py`.
- [__`python`__ `transform.py`] Transforms values & column names, assigns class label column.
- [__`python`__ `train_test_split.py`] Prepares train and test data sets stratified by the class of interest and fills nan values in test and train data.
### 2) Feature Selection
- [__`python`__ `feature_selection.py`] Runs chi-square analysis to help identify relevant features.
### 3) Classification Models
- [__`python`__ `logistic_regression.py`] Executes a logit classification of test data.
- [__`python`__ `bayes.py`] Executes a naive Bayesian classification of test data.
- [__`python`__ `decision_tree.py`] Executes a decision tree classification of test data.
- [__`python`__ `random_forest.py`] Executes a random foreset classification of test data.
