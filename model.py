#!/usr/bin/env python
import collections

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from featureSelectionUtils import feature_selection
from SVMutlis import svm_finetuning, svm_best_params, svm_run

import os.path

# read train,val and test from files
X_train = pd.read_csv('input/train.tsv', sep='\t')
X_test = pd.read_csv('input/test.tsv', sep='\t')

# drop target and user_id column
y_train = X_train['target'].copy()
X_train.drop(['target', 'user_id'], axis=1, inplace=True)

y_test = X_test['target'].copy()
X_test.drop(['target', 'user_id'], axis=1, inplace=True)

# if feature selection is implemented import features
if os.path.exists('FSfeatures/features.txt'):
    with open('FSfeatures/features.txt') as f:
        features = f.readlines()
    features = [x.strip() for x in features]

else:
    # returns most significant features
    features = feature_selection(X_train, y_train)

    # store features in file
    with open('FSfeatures/features.txt', 'w') as fp:
        for item in features[1:]:
            # write each item on a new line
            fp.write("%s\n" % item)


print('Feature selection.....Done')

# X_train and X_test after feature selection
X_train = X_train[features]
X_test = X_test[features]


# example dataset
X_train = X_train.iloc[:100]
y_train = y_train.iloc[:100]
X_test = X_test.iloc[:99]
y_test = y_test.iloc[:99]


number_of_folds = 10
kfold = StratifiedKFold(n_splits=number_of_folds, shuffle=True)


# SVM finetuning
svm_dict = collections.defaultdict(lambda: 0.0)
for train_index, test_index in kfold.split(X_train, y_train):
    trainX, valX = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
    trainY, valY = y_train[train_index], y_train[test_index]
    svm_dict = svm_finetuning(X_train, y_train, valX, valY, svm_dict)


# svm_dict = {"[('C', 0.1), ('gamma', 1), ('kernel', 'rbf')]": 8.5}
c, gamma, kernel = svm_best_params(svm_dict, number_of_folds)

# run svm with best parameters and store accuracy
accuracy = svm_run(c, gamma, kernel, X_train, y_train, X_test, y_test)
print('SVM accuracy : ', accuracy)


print('Training done')








