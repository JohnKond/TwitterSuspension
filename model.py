#!/usr/bin/env python


import os.path
import pandas as pd

from featureSelectionUtils import feature_selection
from SaveLoadUtils import save_model, load_model
from RFutlis import rf_finetuning, rf_run
from SVMutlis import svm_finetuning, svm_run
from XGButils import xgb_finetuning, xgb_run

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


def svm_model(X_train, y_train):
    # SVM fine-tuning
    svm_clf, svm_cv_score = svm_finetuning(X_train, y_train)

    # run svm model with tuned parameters
    model = svm_run(svm_clf, X_train, y_train)

    # save model in file
    save_model('SVM', model)
    print('Done SVM classifier')


def random_forest_model(X_train, y_train):
    # Random Forest fine-tuning
    rf_clf, rf_cv_score = rf_finetuning(X_train, y_train)

    # run rf model with tuned parameters
    model = rf_run(rf_clf, X_train, y_train)

    # save model in file
    save_model('RF', model)
    print('Done RandomForest classifier')


def xgb_model(X_train, y_train):
    # XGBoost fine-tuning
    xgb_clf, xgb_cv_score = xgb_finetuning(X_train, y_train)

    # run xgb model with tuned parameters
    model = xgb_run(xgb_clf, X_train, y_train)

    # save model in file
    save_model('XGB', model)
    print('Done XGBoost classifier')


'''

Run models:
1. SVM
2. Random-Forest
3. XGBoost

'''


# svm_model(X_train, y_train)
random_forest_model(X_train, y_train)
# xgb_model(X_train, y_train)


print('Training done')
