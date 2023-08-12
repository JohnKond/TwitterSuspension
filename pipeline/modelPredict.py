"""
-------------------------------
Author : Giannis Kontogiorgakis
Email : csd3964@csd.uoc.gr
-------------------------------
Main model file that manages the model selection, fit, train and predict.
"""

import os.path
import sys
import pandas as pd
from SaveLoadUtils import save_model, load_model, save_scaler, load_scaler
from featureSelectionUtils import import_features
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler

class ModelPredict:
    
    def __init__(self, period,train_input_folder, in_month , balance):
        """
        Initializes an instance of the ModelPredict class.

        Args:
            period (str): The time period under consideration.
            train_input_folder (str): Path to the folder containing training data.
            in_month (bool): True if predicting for the same month as training, False otherwise.
            balance (bool): True to balance the dataset before prediction.
        """
        self.period = period
        self.in_month = in_month
        self.balance = balance
        self.folder_path = train_input_folder
        self.main()

    
    def read_month(self):
        """
        Reads and prepares the data for prediction based on the specified month and parameters.
        """
        print('reading {} month data'.format(self.period))

        if self.in_month:
            print('Predict for all users on month {}'.format(self.period))
            if os.path.isfile('{}{}/test.tsv'.format(self.folder_path, self.period)):
                self.X_test = pd.read_csv('{}{}/test.tsv'.format(self.folder_path, self.period), sep='\t', dtype={"user_id": "string"})
                self.y_test = self.X_test['target'].copy()
                self.X_test.drop(['target'], axis=1, inplace=True)
            else:
                print('Error: test.tsv does not exist. Please run dataSplit.py on period {} first.')
                sys.exit()
        else:
            self.X_test = pd.read_csv('{}{}/selected_users_{}.tsv'.format(self.folder_path, self.period, self.period), sep='\t',dtype={"user_id":"string"})
            self.y_test = self.X_test['target'].copy()
            self.X_test.drop(['target','user_id'],axis=1, inplace=True)

            # balance dataset
            if self.balance == True:
                undersample = RandomUnderSampler(sampling_strategy='majority')
                self.X_test, self.y_test = undersample.fit_resample(self.X_test, self.y_test)


        ''' select features '''
        features = import_features()
        self.X_test = self.X_test[features]
        


    def import_model(self):
        """
        Imports the trained model and scaler for prediction.
        """
        if os.path.isfile('model.pkl'):
            self.model = load_model()
            self.scaler = load_scaler()
        else:
            print('Error: model does not exist')
            sys.exit()
        
    def predict(self, X, y):
        """
        Predicts user suspension scores using the loaded model and scaler.

        Args:
            X (pd.DataFrame): Input features for prediction.
            y (pd.Series): Actual target values.

        """
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        score = f1_score(y_pred, y)
        print('Predict {} user suspension score: {}'.format(self.period, score))


    def main(self):
        """
        The main execution function of the class.
        """
        self.import_model()
        self.read_month()
        self.predict(self.X_test, self.y_test)
        print('End prediction')


