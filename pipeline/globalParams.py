import numpy as np

"""Fine-tune parameters for Logistic-Regression model"""
LR_params = {
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [2000, 3000, 4000, 5000],
    'C': [1000, 100, 10, 1.0, 0.1, 0.01, 0.001],
    'multi_class': ['ovr']
}

"""Fine-tune parameters for Naive-Bayes model"""
NB_params = {
    'var_smoothing': [float(x) for x in np.logspace(0, -9, num=100)]
}

"""Fine-tune parameters for SVM model"""
SVM_params = {
    'C': [0.1, 1, 10, 100, 200, 300, 500],
    'gamma': [1, 0.1, 0.5, 0.01, 0.001, 0.0001],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}


"""Fine-tune parameters for Random Forest model"""
RF_params = {
    "n_estimators": [5, 20, 50, 100, 500, 1000, 1500],  # number of trees in the random forest
    "max_features": ['sqrt', 'log2'],  # number of features in consideration at every split
    "max_depth": [int(x) for x in np.linspace(3, 50, num=12)],  # maximum number of levels allowed in each decision tree
    "min_samples_split": [2, 4, 6, 8, 10],  # minimum sample number to split a node
    "min_samples_leaf": [1, 3, 4],  # minimum sample number that can be stored in a leaf node
    "bootstrap": [True, False]  # method used to sample data points
}


"""Fine-tune parameters for XGBoost model"""
XGB_params = {
        'n_estimators': [300, 400, 500, 1000, 1500, 2000, 2500],
        'colsample_bytree': [0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        'max_depth': [6, 7, 8, 9, 10, 11, 12],
        'gamma': [0, 0.25, 0.5, 1.0],
        'reg_alpha': [0, 0.1, 0.2, 0.3,  0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'reg_lambda': [0, 0.1, 0.2, 0.3,  0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'subsample': [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        'learning_rate': [0.005, 0.01, 0.015],
        'objective': ['binary:logistic', 'multi:softmax', 'multi:softprob']
}

