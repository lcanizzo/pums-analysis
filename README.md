# PUMS Data Analysis

_(supports survey years 2012 - 2019)_

This set of scripts and dictonaries facilitate download and analysis of Census Bureau Public Use Microdata. The currentc configuration seeks to answer a binary classification problem in identifying whether an individual makes more than fourty thousand dollars a year.

The complete PUMs data on persons contains some 500 individual columns of initial responses, recodes, and weights. The `configs/col_name_map.csv` defines the reduced set of columns included in analaysis prior to performing feature selection.

# Configuration
This project is based on a large dataset. To maintain a managable project size, the downloaded data is not included in uploads. Executions were run with the following configuration defined in `_constants.py`
```python
# dictionaries prepared for 2013 - 2019
YEARS = [*range(2013, 2020, 1)]

STATES = ['ak', 'al', 'ar', 'az', 'ca', 'co', 'ct', 'dc', 'de', 'fl', 'ga',
          'hi', 'ia', 'id', 'il', 'in', 'ks', 'ky', 'la', 'ma', 'md', 'me',
          'mi', 'mn', 'mo', 'ms', 'mt', 'nc', 'nd', 'ne', 'nh', 'nj', 'nm',
          'nv', 'ny', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx',
          'ut', 'va', 'vt', 'wa', 'wi', 'wv', 'wy']

# type options: 'p' or 'h'
TYPES = ['p']

COMBINATIONS = np.array(np.meshgrid(YEARS,STATES,TYPES)).T.reshape(-1,3)

# if defined pulls tail = length from each survey, else uses complete dataset.
MAX_LENGTH = 1000
```

The `TYPES` and `COMBINATIONS` declarations should not be modified, but to speed up executions and run through a small example of the analysis, perform the following:
1. Set `YEARS` to a list containing a single year between 2013 and 2019 (ex: `[2019]`).
    - _(this defines the years of data included)_
2. Set `STATES` to a list containing a single lower case state abreviation (ex: `['tx']`).
    - _(this defines the states included)_
3. (optional) If including multiple states or years, set `MAX_LENGTH` to a number between 100 and 1000.
    - _(this defines the maximum length of a an individual states response set for any year in `YEARS`)_

# Execution Sequence:
## 0) Download Data
- [__`python`__ `get_data.py`] Downloads data for years, states, and categories outlined in `_constants.py`.
    - **NOTE:** The default values in `_constants.py` downloads all data currently configured to be read (**18.7 GB**)

## 1) Data Preparation & Review
- [__`python`__ `transform.py`] Transforms values & column names, assigns class label column.
    - Takes the raw data downloaded from the census burea and first reduces the included columns (initial some 513) to a smaller set of the 53 defined in `configs/col_name_map.csv` while updating column names to be human-readable.
    - Assigns values based on column encodings as defined in the included dictionaries from the census burea.
    - Produces a label for the target class `income_over_50k` based on the `UnadjustedTotalPersonIncome` column.
    - Bins `AvgHoursWorkedPerWeek` in a new `hours_worked` column.
    - Stores staged data with and without the class value (`UnadjustedTotalPersonIncome`) for use in classification and visualizations.

## 2) Classification Models

All models in this section include the following:    
1. Preparing train/test splits of staged data.
2. Defining an execution pipeline for performing imputation, encoding, and feature selection (preprocessing).
3. Defines a classifier step in classifier the pipeline(s).
4. Executes and prints metrics on the model.

### Models:
- [__`python`__ `logit.py`] Executes a logit classification of test data.
- [__`python`__ `bayes.py`] Executes a naive Bayesian classification of test data.
- [__`python`__ `tree.py`] Executes a decision tree classification of test data.
- [__`python`__ `random_forest.py`] Executes a random foreset classification of test data.
    - This file will prompt for whether to run hyper-parameter evaluation (0) or the defined model (1). _(Note: if `_constants.py` has been modified, the defined params selected may not be optimal)_.
