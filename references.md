## US census (PUMS)
- https://www2.census.gov/programs-surveys/acs/data/pums/

---

## Python multiprocessing: 
- https://superfastpython.com/multiprocessing-pool-python/#Step_4_Shutdown_the_Process_Pool

---

## Train test split size:
- https://scholarworks.utep.edu/cs_techrep/1209/#:~:text=Empirical%20studies%20show%20that%20the,of%20the%20data%20for%20training (70 train, 30 test)

---

## Binning:
- https://towardsdatascience.com/data-preprocessing-with-python-pandas-part-5-binning-c5bd5fd1b950

---

## Categorical imputation
- https://pypi.org/project/sklearn-pandas/1.5.0/

---

## Categorical feature selection
- https://perso.uclouvain.be/michel.verleysen/papers/kdir11gd.pdf
    - "A lot of features are thus typically
gathered for a specific problem while many of them
can be either redundant or irrelevant. These useless
features often tend to decrease the performances of
the learning (classification or regression) algorithms
(Guyon and Elisseeff, 2003) and slower the whole
learning process. Moreover, reducing the number of
attributes leads to a better interpretability of the problem and of the models, which is of crucial importance
in many industrial and medical applications. Feature
selection thus plays a major role both from a learning
and from an application point of view"

- https://machinelearningmastery.com/feature-selection-with-categorical-data/

### Chi-square:
- https://machinelearningmastery.com/chi-squared-test-for-machine-learning/
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html

---

## Categorical data encoding
- https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py
- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder
- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#sklearn.preprocessing.OrdinalEncoder
- Python Data Science Handbook (see notes)
- LabelEncoder V. OneHotEncoder:
    - https://towardsdatascience.com/choosing-the-right-encoding-method-label-vs-onehot-encoder-a4434493149b

## Continuous data correlation
- https://www.andrews.edu/~calkins/math/edrm611/edrm05.htm 
    - Correlation coefficients whose magnitude are between 0.3 and 0.5 indicate variables which have a low correlation

## Categorical distance clustering (KMedoids)
- https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
- https://medium.com/analytics-vidhya/the-ultimate-guide-for-clustering-mixed-data-1eefa0b4743b
- gower distance:
    - https://pypi.org/project/gower/

---

## Multi-core machine learning Python
- https://machinelearningmastery.com/multi-core-machine-learning-in-python/
    - In a pool you can't use multiple cores, but for runs of your synchronous train and eval models it can be used

---

## Catesian Product
- https://docs.python.org/3/library/itertools.html#itertools.combinations

---

## Logistic Regression (Sci-kit learn solvers)
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
    - solver{‘lbfgs’, ‘liblinear’, ‘newton-cg’, ‘newton-cholesky’, ‘sag’, ‘saga’}, default=’lbfgs’
    Algorithm to use in the optimization problem. Default is ‘lbfgs’. To choose a solver, you might want to consider the following aspects:
    - For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones;
    - For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss;
    - ‘liblinear’ is limited to one-versus-rest schemes.
    - ‘newton-cholesky’ is a good choice for n_samples >> n_features, especially with one-hot encoded categorical features with rare categories. Note that it is limited to binary classification and the one-versus-rest reduction for multiclass classification. Be aware that the memory usage of this solver has a quadratic dependency on n_features because it explicitly computes the Hessian matrix.

## Scikit learn Pipeline
- https://scikit-learn.org/0.16/modules/generated/sklearn.pipeline.Pipeline.html