"""
-------------------------------
Author : Giannis Kontogiorgakis
Email : csd3964@csd.uoc.gr
-------------------------------
Data split file, manages the balance of the dataset and split in train and test files
"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler


parser = argparse.ArgumentParser()
parser.add_argument('--period', type=str, choices=['feb_mar', 'feb_apr', 'feb_may', 'feb_jun'], help='Choose specific period dataset to split')
args = parser.parse_args()


class DataSplit:
    def __init__(self, period):
        self.data_path = "/Storage/gkont/model_input/"
        self.period = period
        assert self.period in ['feb_mar', 'feb_apr', 'feb_may', 'feb_jun']

        self.main()


    def read_month(self):
        """ file path of social (graph) features tsv file """
        filename = self.data_path + '{}/social_features_{}.tsv'.format(self.period, self.period)

        """ read file as dataframe """
        self.df = pd.read_csv(filename, sep='\t', dtype={"user_id": "string"})

        """ drop target column """
        self.Y = self.df["target"].copy()
        self.df.drop(["target", "user_id"], axis=1, inplace=True)


    def balance(self):
        """ balance dataset """
        print('balance dataset')
        undersample = RandomUnderSampler(sampling_strategy='majority')
        self.df_bal, self.Y_bal = undersample.fit_resample(self.df, self.Y)



    def split(self):
        """ Split balanced dataset with stratified train-test split"""
        X_train, X_test, y_train, y_test = train_test_split(df_bal, Y_bal, test_size=0.2, stratify=Y_bal)
        self.store_files(X_train, X_test, y_train, y_test)


    def store_files(self, X_train, X_test, y_train, y_test):
        """ Link target column and store files """
        X_train['target'] = y_train
        X_test['target'] = y_test

        X_train.to_csv('/Storage/gkont/model_input/{}/train.tsv'.format(period), sep='\t')
        X_test.to_csv('/Storage/gkont/model_input/{}/test.tsv'.format(period), sep='\t')


    def main(self):
        self.read_month()
        self.balance()
        self.split()


ds = DataSplit(args.period)










