"""
Tests logistic regression classification on data.
"""
#%%
from sklearn.linear_model import LogisticRegression
from classification_utils import get_data_encoded, print_accuracy, \
    print_confusion_matrix

x_train, x_test, y_train, y_test = get_data_encoded()

log_reg = LogisticRegression(solver='newton-cholesky', n_jobs=-1)
log_reg.fit(x_train, y_train)
y_pred = log_reg.predict(x_test)

print_accuracy(y_pred, y_test)
print_confusion_matrix(y_pred, y_test)
#%%