'''
-------------------------------
Author : Giannis Kontogiorgakis
Email : csd3964@csd.uoc.gr
-------------------------------
Main model file that manages the model selection, fit, train, predict.
'''
import os
import os.path
import pandas as pd
import time
import argparse


from featureSelectionUtils import feature_selection
from SaveLoadUtils import save_params, load_params, save_scores
from modelSelection import ModelSelection
from modelTrain import ModelTrain
from modelPredict import ModelPredict
from modelFit import ModelFit

parser = argparse.ArgumentParser()

parser.add_argument('--period', type=str, help='Specify period for a model procedure.')
parser.add_argument('--model_selection', action='store_true', help='Performs model selectio, hyperparameters tuning and feature selection')
parser.add_argument('--fit', action='store_true', help='Fit model with train data.')
parser.add_argument('--train', action='store_true', help='Model train on specified period data')
parser.add_argument('--predict', action='store_true', help='Model predict on specified period data')
parser.add_argument('--in_month', action='store_true', help=' Used for train_predict procedure. If true, ')
parser.add_argument('--train_predict', action='store_true', help='Model train and predict on a specified period')


args = parser.parse_args()


# change train_input_folder to your folder path that contains train.tsv and test.tsv
train_input_folder = '/Storage/gkont/model_input/'

months = ['feb_mar', 'feb_apr', 'feb_may', 'feb_jun']
period = args.period
assert args.period in months, "Select a valid month period"



'''
    Read input files train.tsv and test.tsv. For this step first
    run dataSplit.py
'''
def read_input():
    # read train and test from files
    X_test = pd.read_csv(train_input_folder + period + '/test.tsv', sep='\t')
    X_train = pd.read_csv(train_input_folder + period + '/train.tsv', sep='\t')

    # drop target column
    y_train = X_train['target'].copy()
    X_train.drop(['target'], axis=1, inplace=True)

    y_test = X_test['target'].copy()
    X_test.drop(['target'], axis=1, inplace=True)
    
    return X_train, y_train, X_test, y_test


def main():


    if args.model_selection:
        ''' read input training and test datasets '''
        print('Reading input files..')
        X_train, y_train, X_test, y_test = read_input()

        ''' perform feature selection and return new train and test dataset '''
        print('Initiating feature selection..')
        X_train, y_train, X_test, y_test = feature_selection(X_train, y_train, X_test, y_test)

        print('Initiating model selection..')
        ''' model selection with training / validation on 1st month data '''
        ModelSelection('feb_mar', X_train, y_train, k_folds=5)

    elif args.train:
        print('Model train on month {}'.format(args.period))
        ''' train/save best model and scaler on specified month data '''
        ModelTrain(train_input_folder, args.period)

    elif args.predict:
        assert args.period in months, "Select a valid month period"
        print('Model predict on month {}'.format(args.period))
        ''' model predict with data from specified months '''
        ModelPredict(args.period, args.in_month, train_input_folder, balance=True)

    elif args.fit:
        assert args.period in months, "Select a valid month period"
        print('Model fit on month {}'.format(args.period))
        ModelFit(args.period, args.in_month, train_input_folder, balance=True)

    elif args.train_predict:
        assert args.period in months, "Select a valid month period"
        print('Train model and predict on train/test month {}'.format(period))
        ModelTrain(train_input_folder, args.period)
        ModelPredict(args.period, train_input_folder, in_month=True, balance=True)

main()
