"""
Test decision tree classification on data.
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from classification_utils import get_data_encoded, print_accuracy, \
    print_confusion_matrix
from process_timer import time_execution

def main():
    x_train, x_test, y_train, y_test = get_data_encoded()
    tree = DecisionTreeClassifier()
    y_pred = tree.fit(x_train, y_train).predict(x_test)

    print_accuracy(y_pred, y_test)
    print_confusion_matrix(y_pred, y_test, 'Decision Tree')
    
if __name__ == '__main__':
    time_execution(main)
#%%