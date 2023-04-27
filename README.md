# PUMS Data Analysis

_(supports survey years 2012 - 2019)_

This set of scripts and dictonaries facilitate download and analysis of Census Bureau Public Use Microdata. The currentc configuration seeks to answer a binary classification problem in identifying whether an individual makes less than twenty thousand dollars a year.

The complete PUMs data on persons contains some 500 individual columns of initial responses, recodes, and weights. The `configs/col_name_map.csv` defines the reduced set of columns included in analaysis prior to performing feature selection.

## Execution Sequence:
### 1) Data Preparation
- [__`python`__ `get_data.py`] Downloads data for years, states, and categories outlined in `_constants.py`.
- [__`python`__ `transform.py`] Transforms values & column names, assigns class label column.

### 3) Classification Models
- [__`python`__ `logit.py`] Executes a logit classification of test data.
- [__`python`__ `bayes.py`] Executes a naive Bayesian classification of test data.
- [__`python`__ `tree.py`] Executes a decision tree classification of test data.
- [__`python`__ `random_forest.py`] Executes a random foreset classification of test data.
