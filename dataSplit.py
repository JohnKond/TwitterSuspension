#!/usr/bin/env python

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


df = pd.read_csv('input/social_features.tsv', sep='\t')

# select target column
Y = df["target"].copy()
df.drop(["target"], axis=1, inplace= True)

# stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.2, stratify=Y)

# link target to dataset
X_train['target'] = y_train
X_test['target'] = y_test

# store train and test dataset in files
X_train.to_csv('input/train.tsv', sep='\t')
X_test.to_csv('input/test.tsv', sep='\t')






