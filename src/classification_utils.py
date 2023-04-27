"""
Common utilities for classification files.
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

bool_cols = ['income_over_40k']

def get_categorical_cols(df):
   """
   Given a dataframe, returns a list of columns that are categorical.
   """
   return df.select_dtypes(['object','category']).columns

def get_continuous_cols(df):
   """
   Given a dataframe, returns a list of columns that are continuous.
   """
   bool_cols = ['income_over_40k']
   cat_cols = get_categorical_cols(df)
   columns = df.columns
   return [
       col for col in columns if col not in cat_cols and col not in bool_cols
    ]

def print_accuracy(y_pred, y_test):
    """
    Prints accuracy score.
    """

def print_confusion_matrix(y_pred, y_test, model_name=''):
    """
    Prints confusion_matrix results and plots display.
    """
    [[tp, fn], [fp, tn]] = \
        confusion_matrix(y_test, y_pred)
    print(f'TP: {tp}')
    print(f'FP: {fp}')
    print(f'TN: {tn}')
    print(f'FN: {fn}')
    print(f'TPR: {round(tp / (tp + fn),5)}')
    print(f'TNR: {round(tn / (tn + fp),5)}')
    print('\n')
    precision = round(tp / (tp + fp), 5)
    recall = round(tp / (tp + fn),5)
    f1 = round(2 * ((precision * recall)/(precision + recall)), 5)
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'f1: {f1}')
    print('\n')
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

def print_metrics(y_pred, y_test, model_name=''):
    """
    Prints: accuracy, confusion matrix, precision, recall, and f1 scores.
    """
    # Print accuracy
    accuracy = round(accuracy_score(y_test, y_pred), 3)
    print('\nPrediction accuracy: ', accuracy)

    # Print confusion matrix, TPR (Recall), TNR, Precision, and f1.
    [[tp, fn], [fp, tn]] = \
        confusion_matrix(y_test, y_pred)
    print(f'TP: {tp}')
    print(f'FP: {fp}')
    print(f'TN: {tn}')
    print(f'FN: {fn}')
    print(f'TPR: {round(tp / (tp + fn),5)}')
    print(f'TNR: {round(tn / (tn + fp),5)}')
    print('\n')
    precision = round(tp / (tp + fp), 5)
    recall = round(tp / (tp + fn),5)
    f1 = round(2 * ((precision * recall)/(precision + recall)), 5)
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'f1: {f1}')
    print('\n')

    # Plot confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()
#%%