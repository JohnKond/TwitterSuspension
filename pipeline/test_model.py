import json
import os
import os.path
import xgboost as xgb
from sklearn.metrics import f1_score
from SaveLoadUtils import load_params,save_model,load_model,save_scaler,load_scaler
from sklearn.preprocessing import MinMaxScaler

#data_folder = 'C:/Users/giankond/Documents/thesis/Project/data/'
path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/'

class TestModel:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.main()

    def import_model(self):
        print('importing model from file')
        self.model = load_model()

    def train_model(self):
        #f = open(path+'output/params/XGB_params.json', "r")
        #model_params = json.loads(f.read())
        
        model_params = load_params('XGB')

        self.model = xgb.XGBClassifier(
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
            predictor='gpu_predictor',
            tree_method='gpu_hist'
        )

        ''' Scale data '''
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(self.X_train.copy())
        
        ''' Store scaller '''
        save_scaler(scaler)

        ''' Train model with first month trained data '''
        self.model.fit(X_train_scaled, self.y_train)

        ''' Save model '''
        save_model(self.model)
        

   

    def model_predict(self):
        y_pred = self.clf.predict(self.X_test)
        score = f1_score(self.y_test, y_pred)
        print(score)


    def main(self):
        
        if os.path.isfile('model.pkl'):
            print('Importing model from file..')
            self.model = load_model()
            self.scaler = load_scaler()
        else:
            print('Train model with best parameters')
            self.train_model()
            #print('Predict X_test')
            #self.model_predict()

        return None

