import numpy as np
K_folds = 5

"""Fine-tune parameters for Rnadom Forest model"""
RF_params = {"n_estimators": [5, 20, 50, 100, 500, 1000, 1500],  # number of trees in the random forest
             "max_features": ['sqrt', 'log2'],  # number of features in consideration at every split
            "max_depth": [int(x) for x in np.linspace(3, 50, num=12)],  # maximum number of levels allowed in each decision tree
            "min_samples_split": [2, 4, 6, 8, 10],  # minimum sample number to split a node
            "min_samples_leaf": [1, 3, 4],  # minimum sample number that can be stored in a leaf node
            "bootstrap": [True, False]  # method used to sample data points
            }


