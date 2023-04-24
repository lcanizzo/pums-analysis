"""
Test k medoids classification on data.
"""
#%%
import gower
from sklearn_extra.cluster import KMedoids
from classification_utils import get_data_encoded, print_accuracy, print_confusion_matrix
from process_timer import time_execution

def main():
    x_train, x_test, y_train, y_test = get_data_encoded()

    # compute distance matrices
    print('get train gower distance matrix...')
    train_x_dist = gower.gower_matrix(x_train)
    print('get test gower distance matrix...')
    test_x_dist = gower.gower_matrix(x_test)

    # k-medoids classification
    print('fit k-medoids')
    kmedoids = KMedoids(n_clusters=10, random_state=1).fit(train_x_dist, y_train)
    print('predict from x test distances')
    y_pred = kmedoids.predict(test_x_dist)
    print('\nKMedoids y predictions:')
    print(y_pred)

    # Accuracy
    print_accuracy(y_pred, y_test)

    # Confusion matrix
    print_confusion_matrix(y_pred, y_test)

time_execution(main)
#%%