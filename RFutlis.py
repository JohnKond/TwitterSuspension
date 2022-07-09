import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


def rf_finetuning(X_train, y_train, X_test, y_test, rf_dict):
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

    rf = RandomForestRegressor()
    clf = RandomizedSearchCV(estimator=rf, param_distributions=param_grid,
                                   n_iter=100, cv=5, verbose=2, random_state=35, n_jobs=-1)

    clf.fit(X_train, y_train)
    best_params = clf.best_params_
    param_pairs = str(list(best_params.items())).replace('\'','"')
    y_pred = clf.predict(X_test)

    y_pred
    y_test
    accuracy = metrics.accuracy_score(y_test, y_pred)

    rf_dict[param_pairs] += accuracy
    return rf_dict


# return best parameters (C, gamma, kernel)
def rf_best_params(rf_dict, number_of_folds):
    for params in rf_dict:
        rf_dict[params] /= number_of_folds

    rf_dict = {k: v for k, v in sorted(rf_dict.items(), key=lambda item: item[1])}
    best_params = eval(next(iter(rf_dict)))

    # C, C_value = (best_params[0])
    # gamma, gamma_value = (best_params[1])
    # kernel, kernel_value = (best_params[2])

    return


# run svm with parameters C,gamma, kernel and returns accuracy
def rf_run(c, gamma, kernel, X_train, y_train, X_test, y_test):
    # svm classifier
    clf = svm.SVC(kernel=kernel, C=c, gamma=gamma)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy

