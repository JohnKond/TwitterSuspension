import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


def rf_finetuning(X_train, y_train):
    print('RandomForest finetuning')
    n_estimators = [5, 20, 50, 100]  # number of trees in the random forest
    max_features = ['auto', 'sqrt']  # number of features in consideration at every split
    max_depth = [int(x) for x in np.linspace(10, 120, num=12)]  # maximum number of levels allowed in each decision tree
    min_samples_split = [2, 6, 10]  # minimum sample number to split a node
    min_samples_leaf = [1, 3, 4]  # minimum sample number that can be stored in a leaf node
    bootstrap = [True, False]  # method used to sample data points

    param_grid = {'n_estimators': n_estimators,
                  'max_features': max_features,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'bootstrap': bootstrap}

    cv = StratifiedKFold(n_splits=10, shuffle=True)  # 10 folds
    n_jobs = -1  # all processors to be used
    scoring = 'f1'  # f1 scoring function
    rf = RandomForestClassifier()

    clf = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=100,
        cv=cv,
        verbose=3,
        scoring=scoring,
        random_state=35,
        n_jobs=n_jobs,
        return_train_score=True)

    clf.fit(X_train, y_train)
    results = clf.cv_results_
    print(clf.best_params_)
    print(clf.best_score_)
    return clf.best_params_, clf.best_score_


# run RandomForest classifier with tunes parameters
def rf_run(params, X_train, y_train):
    n_estimators = params['n_estimators']
    max_features = params['max_features']
    max_depth = params['max_depth']
    min_samples_split = params['min_samples_split']
    min_samples_leaf = params['min_samples_leaf']
    bootstrap = params['bootstrap']

    # rf classifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap
    )
    clf.fit(X_train, y_train)
    return clf
