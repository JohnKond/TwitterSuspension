"""
-------------------------------
Author : Giannis Kontogiorgakis
Email : csd3964@csd.uoc.gr
-------------------------------
modelParamsUtils.py is responsible for creating the parameters list for each model, for fine-tuning.
"""
import ast
import random
from globalParams import SVM_params, RF_params, XGB_params, NB_params, LR_params

'''
Create parameter list for Linear Regression model
'''
def LR_parameter_list():
    params = []
    for solver in LR_params['solver']:
        for max_iter in LR_params['max_iter']:
            for C in LR_params['C']:
                for multi_class in LR_params['multi_class']:
                    params.append({"solver": solver, "max_iter": max_iter, "C": C, "multi_class": multi_class})
    random.shuffle(params)
    # config = random.sample(params, 100)
    return params


'''
Create parameter list for Naive Bayes model
'''
def NB_parameter_list():
    params = []
    for var in NB_params['var_smoothing']:
        params.append({"var_smoothing":var})
    random.shuffle(params)
    # config = random.sample(params, 100)
    return params

'''
Create parameter list for SVM model
'''
def SVM_parameter_list():
    params = []
    for c in SVM_params['C']:
        for gamma in SVM_params['gamma']:
            for kernel in SVM_params['kernel']:
                params.append( {"C":c,"gamma":gamma,"kernel":kernel})
    random.shuffle(params)
    config = random.sample(params,100)    
    return config

'''
Create parameter list for Random Forest model
'''
def RF_parameter_list():
    params = []
    for n_estimators in RF_params['n_estimators']:
        for max_features in RF_params['max_features']:
            for max_depth in RF_params['max_depth']:
                for min_samples_split in RF_params['min_samples_split']:
                    for min_samples_leaf in RF_params['min_samples_leaf']:
                        for bootstrap in RF_params['bootstrap']:
                            params.append({"n_estimators" : n_estimators,
                                            "max_features" : max_features,
                                            "max_depth" : max_depth,
                                            "min_samples_split": min_samples_split,
                                            "min_samples_leaf": min_samples_leaf,
                                            "bootstrap": bootstrap
                                            })
   
    random.shuffle(params)
    config = random.sample(params,100)    
    return config

'''
Create parameter list for XGBoost model
'''
def XGB_parameter_list():
    params = []
    for n_estimators in XGB_params['n_estimators']:
        for colsample_bytree in XGB_params['colsample_bytree']:
            for max_depth in XGB_params['max_depth']:
                for gamma in XGB_params['gamma']:
                    for reg_alpha in XGB_params['reg_alpha']:
                        for reg_lambda in XGB_params['reg_lambda']:
                            for subsample in XGB_params['subsample']:
                                for learning_rate in XGB_params['learning_rate']:
                                    for objective in XGB_params['objective']:
                                        params.append({"n_estimators" : n_estimators,
                                                        "colsample_bytree" : colsample_bytree,
                                                        "max_depth" : max_depth,
                                                        "gamma" : gamma,
                                                        "reg_alpha": reg_alpha, 
                                                        "reg_lambda": reg_lambda,
                                                        "subsample": subsample, 
                                                        "learning_rate": learning_rate,
                                                        "objective" : objective
                                                        })

    random.shuffle(params)
    config = random.sample(params,100)
    return config
