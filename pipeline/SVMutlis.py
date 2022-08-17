from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from global_params import K_folds, SVM_params


def svm_finetuning(X_train, y_train):
    print('SVM finetuning')

    clf = GridSearchCV(
        estimator=SVC(),
        param_grid=SVM_params,
        scoring='f1',
        cv=StratifiedKFold(n_splits=K_folds, shuffle=True),
        refit=True,
        verbose=3,
        n_jobs=-1,
        return_train_score=True)

    clf.fit(X_train, y_train)

    print(clf.best_score_)
    print(clf.best_params_)

    return clf.best_params_, clf.best_score_


# run svm with parameters C,gamma, kernel and returns accuracy
def svm_run(clf, X_train, y_train):
    clf.fit(X_train, y_train)
    return clf



