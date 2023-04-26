"""
Tests naive Bayes classification on data.
"""
#%%
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from classification_utils import get_data_encoded, print_accuracy, \
    print_confusion_matrix
from process_timer import time_execution

def main():
    x_train, x_test, y_train, y_test = get_data_encoded()

    # Gaussian Naive Bayes
    gnb = GaussianNB()
    y_pred = gnb.fit(x_train, y_train).predict(x_test)

    print('\nGaussian Naive Bayes predictions:')
    print(y_pred)

    print_accuracy(y_pred, y_test)
    print_confusion_matrix(y_pred, y_test, 'Gaussian Naive Bayes')

    # Bernoulli Naive Bayes
    bnb = BernoulliNB()
    y_pred = bnb.fit(x_train,y_train).predict(x_test)

    print('\nBernoulli Naive Bayes predictions:')
    print(y_pred)

    print_accuracy(y_pred, y_test)
    print_confusion_matrix(y_pred, y_test, 'Bernoulli Naive Bayes')

    # Multinomial Naive Bayes
    mnb = MultinomialNB()
    y_pred = mnb.fit(x_train, y_train).predict(x_test)

    print('\nMultinomial Naive Bayes predictions:')
    print(y_pred)

    print_accuracy(y_pred, y_test)
    print_confusion_matrix(y_pred, y_test, 'Multinomial Naive Bayes')


if __name__ == '__main__':
    time_execution(main)
#%%