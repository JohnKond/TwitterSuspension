#!/usr/bin/env python

import pandas as pd
import numpy as np
import collections

from sklearn.linear_model import Lasso
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error


def Lasso_finetuning(X_train, Y_train, X_test, Y_test, lasso_dict):
    alpha = np.arange(0.0001, 0.1, 0.001)

    for a_param in alpha:
        lasso = Lasso(max_iter=10000, alpha=a_param)
        lasso.fit(X_train, Y_train)
        mse = mean_squared_error(Y_test, lasso.predict(X_test))
        lasso_dict[a_param] += mse

    return lasso_dict


def select_best_alpha(lasso_dict, number_of_folds):
    for alpha in lasso_dict:
        lasso_dict[alpha] /= number_of_folds

    lasso_dict = {k: v for k, v in sorted(lasso_dict.items(), key=lambda item: item[1])}
    alpha = next(iter(lasso_dict))
    return alpha


def significant_features(X_train, y_train, alpha):
    features = []
    lasso = Lasso(max_iter=10000, alpha=alpha)
    lasso.fit(X_train, y_train)
    ser = pd.Series(lasso.coef_, index=X_train.columns)
    for i, v in ser.items():
        if abs(v) != 0:
            features.append(i)
    return features


# perform feature selection on train dataset
def feature_selection(X_train, y_train):
    number_of_folds = 10
    kfold = StratifiedKFold(n_splits=number_of_folds, shuffle=True)

    lasso_dict = collections.defaultdict(lambda: 0.0)
    for train_index, test_index in kfold.split(X_train, y_train):
        trainX, valX = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
        trainY, valY = y_train[train_index], y_train[test_index]
        lasso_dict = Lasso_finetuning(trainX, trainY, valX, valY, lasso_dict)

    # find the best alpha param
    alpha = select_best_alpha(lasso_dict, number_of_folds)

    return significant_features(X_train, y_train, alpha=alpha)

