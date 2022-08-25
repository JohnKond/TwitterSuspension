#!/usr/bin/env python

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

period = 'feb_mar'
filename = '/Storage/gkont/embeddings/{}/social_features_{}.tsv'.format(period,period)


df = pd.read_csv(filename, sep='\t', dtype={"user_id":"string"})

# select target column
Y = df["target"].copy()
df.drop(["target","user_id"], axis=1, inplace= True)


# balance dataset
print('balance dataset')
undersample = RandomUnderSampler(sampling_strategy='majority')
df_bal, Y_bal = undersample.fit_resample(df, Y)

print('Y_balanced value counts: ', Y_bal.value_counts())

# stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(df_bal, Y_bal, test_size=0.2, stratify=Y_bal)


# link target to dataset
X_train['target'] = y_train
X_test['target'] = y_test

# store train and test dataset in files
X_train.to_csv('/Storage/gkont/model_input/{}/train.tsv'.format(period), sep='\t')
X_test.to_csv('/Storage/gkont/model_input/{}/test.tsv'.format(period), sep='\t')

print('Stored train and test dataset')



def read_month():
    return None








