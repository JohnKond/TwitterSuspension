import os.path
import sys

import pandas as pd
from SaveLoadUtils import save_model, load_model, save_scaler, load_scaler
from featureSelectionUtils import import_features
from imblearn.under_sampling import RandomUnderSampler

class ModelFit:

    def __init__(self, period, in_month, train_folder, balance, ):
        self.period = period
        self.in_month = in_month
        self.folder_path = train_folder
        self.balance = balance
        self.main()


    def import_model(self):
        print('Importing model and scaler')
        self.model = load_model()
        self.scaler = load_scaler()


    def read_month(self):
        #print('read previous users of {}'.format(self.period))

        if self.in_month:
            if os.path.isfile('{}{}/train.tsv'.format(self.folder_path, self.period)):
                self.X_train = pd.read_csv('{}{}/train.tsv'.format(self.folder_path, self.period), sep='\t', dtype={"user_id": "string"})
                self.y_train = self.X_train['target'].copy()
                self.X_train.drop(['target'], axis=1, inplace=True)
            else:
                print('Error: train.tsv does not exist. Please run dataSplit.py on period {} first.')
                sys.exit()
        else:
            self.X_train = pd.read_csv('{}{}/previous_users_{}.tsv'.format(self.folder_path, self.period, self.period), sep='\t', dtype={"user_id": "string"})
            self.y_train = self.X_train['target'].copy()
            self.X_train.drop(['target', 'user_id'], axis=1, inplace=True)


        # select features
        features = import_features()
        self.X_train = self.X_train[features]



        # balance dataset
        if self.balance == True:
            undersample = RandomUnderSampler(sampling_strategy='majority')
            self.X_train, self.y_train = undersample.fit_resample(self.X_train, self.y_train)


    def fit_model(self, X, y):

        # scale data
        X_scaled = self.scaler.fit_transform(X.copy()) # transform or fit_transform
        self.model.fit(X_scaled, y)
        save_model(self.model)
        save_scaler(self.scaler)


    def main(self):
        self.import_model()
        self.read_month()
        self.fit_model(self.X_train, self.y_train)
        print('End fit model')
