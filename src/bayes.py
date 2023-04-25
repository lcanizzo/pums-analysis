"""
Tests naive Bayes classification on data.
"""
#%%
import pandas as pd
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from classification_utils import get_data_encoded, print_accuracy, \
    print_confusion_matrix

x_train, x_test, y_train, y_test = get_data_encoded()

# Gaussian Naive Bayes
gnb = GaussianNB()
y_pred = gnb.fit(x_train, y_train).predict(x_test)

print('\nGaussian Naive Bayes predictions:')
print(y_pred)

print_accuracy(y_pred, y_test)
print_confusion_matrix(y_pred, y_test)

# Categorical Naive Bayes
clf = CategoricalNB(min_categories=10)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print('\nCategorical Naive Bayes predictions:')
print(y_pred)

print_accuracy(y_pred, y_test)
print_confusion_matrix(y_pred, y_test)
#%%