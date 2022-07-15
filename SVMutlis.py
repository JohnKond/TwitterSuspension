from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from utils.global_params import K_folds
import pickle


def svm_finetuning(X_train, y_train):
    print('SVM finetuning')
    # GridSearch parameters
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': ['auto', 1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf', 'linear']}

    cv = StratifiedKFold(n_splits=K_folds, shuffle=True)      # 10 folds
    n_jobs = -1      # all processors to be used
    scoring = 'f1'   # scoring function

    clf = GridSearchCV(
        estimator=SVC(),
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        refit=True,
        verbose=3,
        n_jobs=n_jobs,
        return_train_score=True)

    clf.fit(X_train, y_train)

    results = clf.cv_results_
    print(clf.best_params_)
    print(clf.best_score_)
    return clf.best_params_, clf.best_score_


# run svm with parameters C,gamma, kernel and returns accuracy
def svm_run(params, X_train, y_train):
    c = params['C']
    gamma = params['gamma']
    kernel = params['kernel']

    # svm classifier
    clf = svm.SVC(kernel=kernel, C=c, gamma=gamma)
    clf.fit(X_train, y_train)
    return clf



