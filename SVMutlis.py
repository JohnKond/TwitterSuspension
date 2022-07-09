from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC



def svm_finetuning(X_train, y_train, X_test, y_test, svm_dict):

    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf', 'linear']}

    clf = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
    clf.fit(X_train, y_train)
    best_params = clf.best_params_
    param_pairs = str(list(best_params.items()))
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)

    y_pred
    y_test

    svm_dict[param_pairs] += accuracy
    return svm_dict


# return best parameters (C, gamma, kernel)
def svm_best_params(svm_dict, number_of_folds):
    for params in svm_dict:
        svm_dict[params] /= number_of_folds

    svm_dict = {k: v for k, v in sorted(svm_dict.items(), key=lambda item: item[1])}
    best_params = eval(next(iter(svm_dict)))

    C, C_value = (best_params[0])
    gamma, gamma_value = (best_params[1])
    kernel, kernel_value = (best_params[2])

    return C_value, gamma_value, kernel_value


# run svm with parameters C,gamma, kernel and returns accuracy
def svm_run(c, gamma, kernel, X_train, y_train, X_test, y_test):
    # svm classifier
    clf = svm.SVC(kernel=kernel, C=c, gamma=gamma)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    y_pred
    y_test
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy

