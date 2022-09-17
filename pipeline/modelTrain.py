import json
import os
import os.path
import sys

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score
from SaveLoadUtils import load_params,save_model,load_model,save_scaler,load_scaler
from sklearn.preprocessing import MinMaxScaler

#data_folder = 'C:/Users/giankond/Documents/thesis/Project/data/'
from featureSelectionUtils import import_features

path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/'

class ModelTrain:
    def __init__(self, train_folder, period):
        self.folder_path = train_folder
        self.period = period
        self.main()

    def read_month(self):
        if os.path.isfile('{}{}/train.tsv'.format(self.folder_path, self.period)):
            print('Read month {}'.format(self.period))
            self.X = pd.read_csv('{}{}/train.tsv'.format(self.folder_path, self.period), sep='\t', dtype={"user_id": "string"})
            self.y = self.X['target'].copy()
            self.X.drop(['target'], axis=1, inplace=True)

            # select features
            features = import_features()
            self.X = self.X[features]
        else:
            print('Error: train.tsv does not exist. Please run dataSplit.py on period {} first.')
            sys.exit()



    def import_model(self):
        print('importing model from file')
        self.model = load_model()

    def train_model(self, X, y):
        model_params = load_params('XGB')

        self.model = xgb.XGBClassifier(
            n_estimators=model_params['n_estimators'],
            colsample_bytree=model_params['colsample_bytree'],
            max_depth=model_params['max_depth'],
            gamma=model_params['gamma'],
            reg_alpha=model_params['reg_alpha'],
            reg_lambda=model_params['reg_lambda'],
            subsample=model_params['subsample'],
            learning_rate=model_params['learning_rate'],
            objective=model_params['objective'],
            # GPU
            predictor='gpu_predictor',
            tree_method='gpu_hist'
        )

        ''' Scale data '''
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X.copy())
        
        ''' Store scaller '''
        save_scaler(scaler)

        ''' Train model with first month trained data '''
        self.model.fit(X_scaled, y)

        ''' Save model '''
        save_model(self.model)


    def main(self):
        self.read_month()
        if os.path.isfile('model.pkl'):
            print('Model already exists')
        else:
            print('Train model with best parameters')
            self.train_model(self.X, self.y)
            print('Training done')
        return 

