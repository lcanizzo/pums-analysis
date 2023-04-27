"""
Runs random forest tree classification with and testing of hyper-params to
classify test data on whether persons make more than 40k a year.
"""

#%%
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from itertools import product
from multiprocessing import Pool
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, chi2
from classification_utils import print_accuracy, print_confusion_matrix, \
get_categorical_cols, get_continuous_cols
from process_timer import time_execution

CORES = os.cpu_count()
RANDOM_STATE = 43
COLORS = [
    'red',
    'blue',
    'green',
    'orange',
    'purple',
    'lightcoral',
    'cornflowerblue',
    'springgreen',
    'peru',
    'fuchsia',
    'maroon',
    'slategrey',
    'yellowgreen',
    'sienna',
    'plum'   
]

np.random.seed(0)

# Train test split
df = pd.read_csv('./compiled_data/staged/all_transformed.csv')

features_x = df.drop(['income_over_40k'], axis=1)
class_y = df['income_over_40k']
x_train, x_test, y_train, y_test = train_test_split(
    features_x,
    class_y,
    test_size=0.3,
    random_state=0
)

# Prepare imputer, scaler / encoder, and categorical feature selection
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("selector", SelectPercentile(chi2, percentile=25)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, get_continuous_cols(df)),
        ("cat", categorical_transformer, get_categorical_cols(df)),
    ],
    sparse_threshold=0
)


def test_params(params):
    """
    Runs random forest classifier on data and returns
    params n, d, and the error for the run.
    """
    n, d = params
    print(f'\ntest_rndf_params: n="{n}"; d="{d}"' )
    rndf = Pipeline(
        steps=[
            ("preprocessor", preprocessor), 
            ("classifier", RandomForestClassifier(
                max_depth=d,
                n_estimators=n,
                criterion='entropy',
                random_state=RANDOM_STATE
            ))
        ]
    )
    rndf_y_pred = rndf.fit(x_train, y_train).predict(x_test)
    rndf_acc = round(accuracy_score(y_test, rndf_y_pred), 4)
    rndf_err = round(1 - rndf_acc, 4)
    return [n, d, rndf_err]

if __name__ == '__main__':
    def main():
        ## Test Random Forest hyper-parameters
        test_hyper_params = input(
            'Run defined model (1) or test hyper-params (0): '
        ).lower().strip() == '0'

        # If depth and n subtrees need to be tested, run evaluations
        if test_hyper_params:
            # subtrees up to 20 supported
            test_n_subtrees = 20
            # max depth up to 15 supported
            test_depth_to = 15

            subtrees = np.arange(1,test_n_subtrees + 1).tolist()
            print(f'For n subtrees: {subtrees}')
            depths = np.arange(1,test_depth_to + 1).tolist()
            print(f'And d depths: {depths}')

            hyper_params = list(product(subtrees, depths))
            err_rates = pd.DataFrame()

            with Pool() as pool:
                results = pool.map(test_params, hyper_params)
                err_rates = pd.DataFrame(results, columns =['n', 'd', 'err'])

            print('\nError rates dataframe:')
            err_rates.sort_values(by=['d'], inplace=True)
            print(err_rates)

            ax = plt.axes()
            ax.set_xticks(subtrees)
            for i, d in enumerate(depths):
                plt.plot(
                    subtrees,
                    err_rates[err_rates['d'] == d]['err'],
                    color=COLORS[i],
                    label=f'd = {d}'
                )
            plt.xlabel('n subtrees')
            plt.ylabel('Error Rate')
            plt.title('Random Forest hyper-parameter performance')
            plt.legend(
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                borderaxespad=0.
            )
            plt.show()
        # Run single optimized model if not testing hyper-params
        else:
            ## If tuning threshold
            print(
                '\nThe most performant hyper-params for Random Forest are:'
            )
            threshold = 0.5
            n = 20
            d = 15
            print(f'n = {n}')
            print(f'd = {d}')
            rndf = Pipeline(
                steps=[
                    ("preprocessor", preprocessor), 
                    ("classifier", RandomForestClassifier(
                       n_jobs=-1,
                        max_depth=d,
                        n_estimators=n,
                        criterion='entropy',
                        random_state=RANDOM_STATE
                    ))
                ]
            )
            rndf.fit(x_train, y_train)
            probabilities = rndf.predict_proba(x_test)
            y_pred = (probabilities [:,1] >= threshold).astype('int')
            print_accuracy(y_pred, y_test)
            print_confusion_matrix(y_pred, y_test, 'Random Forest')
            print(f'threshold: {threshold}')

    time_execution(main)
#%%