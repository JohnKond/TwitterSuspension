import ast
import json
import time
from collections import defaultdict
import sys
import pickle
import os.path
#sys.path.append("/home/gkont/TwitterSuspension")

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from modelParamsUtils import SVM_parameter_list, RF_parameter_list, XGB_parameter_list
from SaveLoadUtils import save_params, save_scores
# GPU libs (cuML)
# from cuml.svm import SVC as SVC_gpu
# from cuml import RandomForestClassifier as cuRF
#from thundersvm import SVC

#path_to_project = "/home/gkont/TwitterSuspension"

class TrainModel:
    def __init__(self, period, X_train, y_train, k_folds):
        self.period = period
        self.X_train = X_train
        self.y_train = y_train
        self.k_folds = k_folds
        self.folds_dict = {}

        self.main()


    def model_classifier(self, name, entry):
        if name == 'SVM':
            #clf = SVC_gpu(
            clf = svm.SVC(
                          kernel=entry['kernel'],
                          C=float(entry['C']),
                          gamma=float(entry['gamma']),
                          verbose=0)
        elif name == 'RF':
            if entry['bootstrap'] == 'True':
                bootstrap = True
            else:
                bootstrap = False

            #clf = cuRF(
            clf = RandomForestClassifier(
                n_estimators=int(entry['n_estimators']),
                max_features=entry['max_features'],
                max_depth=int(entry['max_depth']),
                min_samples_split=int(entry['min_samples_split']),
                min_samples_leaf=int(entry['min_samples_leaf']),
                bootstrap=bootstrap,
                n_jobs=-1,
                verbose=0
            )
        else:
            if entry['objective'] == 'binary:logistic':
                clf = xgb.XGBClassifier(
                    n_estimators=int(entry['n_estimators']),
                    colsample_bytree=float(entry['colsample_bytree']),
                    max_depth=int(entry['max_depth']),
                    gamma=float(entry['gamma']),
                    reg_alpha=float(entry['reg_alpha']),
                    reg_lambda=float(entry['reg_lambda']),
                    subsample=float(entry['subsample']),
                    learning_rate=float(entry['learning_rate']),
                    objective=entry['objective'],
                    # GPU
                    predictor='gpu_predictor',
                    tree_method='gpu_hist'
                )
            else:
                clf = xgb.XGBClassifier(
                    n_estimators=int(entry['n_estimators']),
                    colsample_bytree=float(entry['colsample_bytree']),
                    max_depth=int(entry['max_depth']),
                    gamma=float(entry['gamma']),
                    reg_alpha=float(entry['reg_alpha']),
                    reg_lambda=float(entry['reg_lambda']),
                    subsample=float(entry['subsample']),
                    learning_rate=float(entry['learning_rate']),
                    objective=entry['objective'],
                    num_class=2,
                    # GPU
                    predictor='gpu_predictor',
                    tree_method='gpu_hist'
                )


        return clf


    def model_config(self, name):
        if name == 'SVM':
            return SVM_parameter_list()
        elif name == 'RF':
            return RF_parameter_list()
        else:
            return XGB_parameter_list()


    def model_train(self, name):
        val_performance = defaultdict(lambda: 0.0)
        train_performance = defaultdict(lambda: 0.0)
        config = self.model_config(name)
        scaller = StandardScaler()
        print('Will test {} setups'.format(len(config)))
        fold_index = 1
        for folds in self.folds_dict.values():
            train_index = folds[0]
            val_index = folds[1]
            train_X, val_X = self.X_train.iloc[train_index, :], self.X_train.iloc[val_index, :]
            train_Y, val_Y = self.y_train.iloc[train_index,], self.y_train.iloc[val_index,]
            
            ''' Scale data '''
            train_X = pd.DataFrame(scaller.fit_transform(train_X.copy()), columns=train_X.columns.to_list())
            val_X = pd.DataFrame(scaller.fit_transform(val_X.copy()), columns=val_X.columns.to_list())

            setup_index = 1
            for entry in config:

                print(' - Testing setup : {} (fold {})'.format(setup_index,fold_index))

                # select classifier
                clf = self.model_classifier(name, entry)

                # Train the model using the training sets
                clf.fit(train_X, train_Y)


                if name == 'XGB' and entry['objective'] == 'multi:softprob':
                    val_Y_pred = clf.predict(val_X).astype(int)
                    train_Y_pred = clf.predict(train_X).astype(int)

                    val_Y_pred = np.argmax(val_Y_pred, axis=1)
                    train_Y_pred = np.argmax(train_Y_pred, axis=1)
                else:
                    # Predict the response for test dataset
                    val_Y_pred = clf.predict(val_X)
                    train_Y_pred = clf.predict(train_X)

                # Score
                f1_val = f1_score(val_Y, val_Y_pred)
                f1_train = f1_score(train_Y, train_Y_pred)

                # store score
                val_performance[str(entry)] += f1_val
                train_performance[str(entry)] += f1_train
                setup_index = setup_index + 1
            print(' [ Fold {} trained ]'.format(fold_index))
            fold_index += 1
        return val_performance, train_performance


    def finetune(self, name):
        print(name+' finetuning')
        print('----------------------\n')


        start_train = time.time()
        performance_val_dict, performance_train_dict = self.model_train(name)
        end_train = time.time()
        training_time = end_train - start_train

        # calculate performance
        best_params = self.best_performance(performance_val_dict)
        best_val_score = performance_val_dict[best_params]/self.k_folds
        best_train_score = performance_train_dict[best_params]/self.k_folds

        print('Training time : ',training_time)
        print('Best params are : ',best_params) 
        print('Best val score is : ', best_val_score)
        print('Best train score is : ', best_train_score)

        ''' Save best parameters and score'''
        save_params(name, ast.literal_eval(best_params))
        save_scores(name, best_val_score, best_train_score ,training_time)

    ''' Given a perfomance dict find average score '''
    def best_performance(self, dict_performance):
        best_param_set = max(dict_performance, key=dict_performance.get)
        return best_param_set

    def models_finetuning(self):
        self.finetune('RF')
        self.finetune('SVM')
        self.finetune('XGB')

    def create_folds(self):
        # check if folds indexes exists
        if os.path.exists('fold_indexes.pkl'):
            print('Importing folds from file..')
            with open('fold_indexes.pkl', "rb") as output:
                self.folds_dict = pickle.load(output)
            return
        
        k_folds = StratifiedKFold(n_splits=self.k_folds, shuffle=True)
        i = 1
        for train_index, val_index in k_folds.split(self.X_train, self.y_train):
            self.folds_dict['fold_'+str(i)] = [train_index, val_index]
            i = i+1

        # store folds indexes in file
        with open('fold_indexes.pkl', "wb") as output:
            pickle.dump(self.folds_dict, output)

        print(str(self.k_folds) + ' folds created..\n')
        

    def main(self):
        self.create_folds()
        self.models_finetuning()
        print('Training finished')
