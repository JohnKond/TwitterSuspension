#!/usr/bin/env python
import os.path
import pandas as pd
import time

import featureSelectionUtils
from SaveLoadUtils import save_params, load_params, save_scores
from RFutlis import rf_finetuning
from SVMutlis import svm_finetuning
from XGButils import xgb_finetuning
from train_model import TrainModel


period = 'feb_mar'
train_input_folder = '/Storage/gkont/model_input/{}/'.format(period)
path = '/home/gkont/TwitterSuspension/'


'''
    Read input files train.tsv and test.tsv. For this step first
    run dataSplit.py
'''
def read_input():
    # read train and test from files
    X_test = pd.read_csv(train_input_folder + 'test.tsv', sep='\t')
    X_train = pd.read_csv(train_input_folder + 'train.tsv', sep='\t')

    # drop target and user_id column
    y_train = X_train['target'].copy()
    X_train.drop(['target'], axis=1, inplace=True)

    y_test = X_test['target'].copy()
    X_test.drop(['target'], axis=1, inplace=True)
    
    return X_train, y_train, X_test, y_test

''' 
    Perform feature selection if not done, and return new 
    train and test files. If feature selection was previously 
    implemented, load the features and return the datasets.
'''
def feature_selection(X_train, y_train, X_test, y_test):

    # if feature selection is implemented import features
    if os.path.exists(path + 'FSfeatures/features.txt'):
        with open(path + '/FSfeatures/features.txt') as f:
            features = f.readlines()
        features = [x.strip() for x in features]

    else:

        fs_start = time.time()
        # returns most significant features
        features = featureSelectionUtils.feature_selection(X_train, y_train)
        fs_end = time.time()
        print('Feature selection time: ', fs_end - fs_start, ' seconds')

        # store features in file
        if not os.path.exists(path + 'FSfeatures'):
            os.mkdir(path + 'FSfeatures')

        with open(path + 'FSfeatures/features.txt', 'w+') as fp:
            for item in features[1:]:
                # write each item on a new line
                fp.write("%s\n" % item)

    print('Feature selection.....Done')

    # X_train and X_test after feature selection
    X_train = X_train[features]
    X_test = X_test[features]

    # example dataset
    #X_train = X_train.iloc[:100]
    #y_train = y_train.iloc[:100]
    #X_test = X_test.iloc[:99]
    #y_test = y_test.iloc[:99]

    return X_train, y_train, X_test, y_test


''' Train SVM model '''
def svm_model(X_train, y_train):

    # SVM fine-tuning
    start_train = time.time()
    svm_best_params, svm_cv_score = svm_finetuning(X_train, y_train)
    end_train = time.time()

    # compute training time
    training_time = end_train - start_train

    # save params and stats
    save_params('SVM', svm_best_params)
    save_scores('SVM', svm_cv_score, training_time)
    print('Done SVM classifier')


''' Train RandomForest model '''
def random_forest_model(X_train, y_train):

    start_train = time.time()
    # Random Forest fine-tuning
    rf_best_params, rf_cv_score = rf_finetuning(X_train, y_train)
    end_train = time.time()

    # compute training time
    training_time = end_train - start_train

    # save model in file
    save_params('RF', rf_best_params)
    save_scores('RF', rf_cv_score, training_time)
    print('Done RandomForest classifier')


''' Train xgb_model '''
def xgb_model(X_train, y_train):

    # XGBoost fine-tuning
    start_train = time.time()
    xgb_best_params, xgb_cv_score = xgb_finetuning(X_train, y_train)
    end_train = time.time()

    # compute training time
    training_time = end_train - start_train

    # save model in file
    save_params('XGB', xgb_best_params)
    save_scores('XGB', xgb_cv_score, training_time)
    print('Done XGBoost classifier')


'''

Run models:
1. SVM
2. Random-Forest
3. XGBoost

'''

def train_models(X_train, y_train):
    svm_model(X_train, y_train)
    random_forest_model(X_train, y_train)
    xgb_model(X_train, y_train)


def main():
    # read input training and test datasets
    print('Reading input files..')
    X_train, y_train, X_test, y_test = read_input()

    # perform feature selection and return new train and test dataset
    print('Initiating feature selection..')
    X_train, y_train, X_test, y_test = feature_selection(X_train, y_train, X_test, y_test)

    # train models
    # train_models(X_train, y_train)
    # TrainModel('feb_mar', X_train, y_train, 5)


main()
print('Training done')
