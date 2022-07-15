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
        scoring='f1',
        random_state=35,
        n_jobs=-1,
        return_train_score=True)

    clf.fit(X_train, y_train)
    results = clf.cv_results_
    print(clf.best_estimator_)
    print(clf.best_score_)
    return clf.best_estimator_, clf.best_score_


# run RandomForest classifier with tuned parameters
def rf_run(clf, X_train, y_train):
    clf.fit(X_train, y_train)
    return clf
