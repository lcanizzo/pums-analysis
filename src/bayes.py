"""
Tests naive Bayes classification on data.
"""
#%%
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from classification_utils import get_data, print_accuracy, print_confusion_matrix

x_train, x_test_enc, y_train, y_test_enc = get_data()

gnb = GaussianNB()
y_pred = gnb.fit(x_train, y_train).predict(x_test_enc)

print('\nGaussian Naive Bayes (GNB) predictions:')
print(y_pred)

print_accuracy(y_pred, y_test_enc)
print_confusion_matrix(y_pred, y_test_enc)
#%%