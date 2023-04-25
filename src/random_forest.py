"""
Tests random forest classification on data.
"""
#%%
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from classification_utils import get_data_encoded, print_accuracy, \
    print_confusion_matrix

RANDOM_STATE = 43
CORES = os.cpu_count()

x_train, x_test, y_train, y_test = get_data_encoded()

def test_params(params):
    n, d = params
    print(f'\ntest_rndf_params: n="{n}"; d="{d}"' )
    rndf = RandomForestClassifier(
        max_depth=d,
        n_estimators=n,
        criterion='entropy',
        random_state=RANDOM_STATE)
    rndf_y_pred = rndf.fit(x_train, y_train).predict(x_test)
    rndf_acc = round(accuracy_score(y_test, rndf_y_pred), 4)
    rndf_err = round(1 - rndf_acc, 4)
    return [n, d, rndf_err]

if __name__ == "__main__":
    from process_timer import time_execution

    def main():
        """
        Uses multiple runs adjusting n & d to determine optimal params for 
        RDF classifier.
        """
        ## Test Random Forest hyper-parameters
        test_hyper_params = False

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

            # print('\n')
            # print('\nBest Random Forest hyper-parameters')
            print('\nError rates dataframe:')
            err_rates.sort_values(by=['d'], inplace=True)
            print(err_rates)

            colors = [
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

            ax = plt.axes()
            ax.set_xticks(subtrees)
            for i, d in enumerate(depths):
                plt.plot(
                    subtrees,
                    err_rates[err_rates['d'] == d]['err'],
                    color=colors[i],
                    label=f'd = {d}'
                )
            plt.xlabel('n subtrees')
            plt.ylabel('Error Rate')
            plt.title('Random Forest hyper-parameter performance')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.show()
        # Else run single optimized model
        else:
            ## If tuning threshold
            threshold = 0.8
            print('\nThe most performant set of hyper-params for Random Forest are:')
            n = 24
            d = 15
            print(f'n = {n}')
            print(f'd = {d}')

            rndf = RandomForestClassifier(
                n_jobs=-1,
                max_depth=d,
                n_estimators=n,
                criterion='entropy',
                random_state=RANDOM_STATE
            )
            rndf.fit(x_train, y_train)
            probabilities = rndf.predict_proba(x_test)
            y_pred = (probabilities [:,1] >= threshold).astype('int')
            print_accuracy(y_pred, y_test)
            print_confusion_matrix(y_pred, y_test)
            print(f'threshold: {threshold}')

    time_execution(main)
#%%