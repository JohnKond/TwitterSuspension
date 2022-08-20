import json
import os
import xgboost as xgb
from sklearn.metrics import f1_score

data_folder = 'C:/Users/giankond/Documents/thesis/Project/data/'
path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/'

class TestModel:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.main()


    def train_model(self):
        f = open(path+'output/params/XGB_params.json', "r")
        model_params = json.loads(f.read())

        self.clf = xgb.XGBClassifier(
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
            # predictor='gpu_predictor',
            # tree_method='gpu_hist'
        )

        self.clf.fit(self.X_train, self.y_train)

    def model_predict(self):
        y_pred = self.clf.predict(self.X_test)
        score = f1_score(self.y_test, y_pred)
        print(score)


    def main(self):
        print('Train model with best paramerters')
        self.train_model()
        print('Predict X_test')
        self.model_predict()

        return None

