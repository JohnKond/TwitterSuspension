#!/usr/bin/env python
import os.path
import pandas as pd
import time

import featureSelectionUtils
from SaveLoadUtils import save_params, load_params, save_scores
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


def main():
    # read input training and test datasets
    print('Reading input files..')
    X_train, y_train, X_test, y_test = read_input()

    # perform feature selection and return new train and test dataset
    print('Initiating feature selection..')
    X_train, y_train, X_test, y_test = feature_selection(X_train, y_train, X_test, y_test)

    # train models
    TrainModel('feb_mar', X_train, y_train,k_folds=5)


main()
print('Training done')