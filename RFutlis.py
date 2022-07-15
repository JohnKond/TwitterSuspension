import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from utils.global_params import K_folds, RF_params


def rf_finetuning(X_train, y_train):
    print('RandomForest finetuning')

    clf = RandomizedSearchCV(
        estimator=RandomForestClassifier(),
        param_distributions=RF_params,
        n_iter=100,
        cv=StratifiedKFold(n_splits=K_folds, shuffle=True),
        verbose=3,
        scoring='f1', # f1 scoring function
        random_state=35,
        n_jobs=-1, # all processors to be used
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
