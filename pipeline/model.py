#!/usr/bin/env python
import os
import os.path
import pandas as pd
import time

from featureSelectionUtils import feature_selection
from SaveLoadUtils import save_params, load_params, save_scores
from modelSelection import ModelSelection
from modelTrain import ModelTrain
from modelPredict import ModelPredict


# change train_input_folder to your folder path that contains train.tsv and test.tsv
period = 'feb_mar'
train_input_folder = '/Storage/gkont/model_input/'

# windows (put in comments)
# train_input_folder = 'C:/Users/giankond/Documents/thesis/Project/data/'

path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/'

'''
    Read input files train.tsv and test.tsv. For this step first
    run dataSplit.py
'''
def read_input():
    # read train and test from files
    X_test = pd.read_csv(train_input_folder + period + '/test.tsv', sep='\t')
    X_train = pd.read_csv(train_input_folder + period + '/train.tsv', sep='\t')

    # drop target and user_id column
    y_train = X_train['target'].copy()
    X_train.drop(['target'], axis=1, inplace=True)

    y_test = X_test['target'].copy()
    X_test.drop(['target'], axis=1, inplace=True)
    
    return X_train, y_train, X_test, y_test


def main():

    # read input training and test datasets
    print('Reading input files..')
    X_train, y_train, X_test, y_test = read_input()

    # perform feature selection and return new train and test dataset
    print('Initiating feature selection..')
    X_train, y_train, X_test, y_test = feature_selection(X_train, y_train, X_test, y_test)

    # print('Initiating model selection..')
    # model selection with training / validation on 1st month data
    # ModelSelection('feb_mar', X_train, y_train, k_folds=5)

    # train/save best model and scaler on first month data
    ModelTrain(X_train, y_train, X_test, y_test)

    # model predict with data from specified months
    # ModelPredict('feb_apr',train_input_folder,fit=True)


main()
