import os.path
import pandas as pd
from SaveLoadUtils import load_model, load_scaler
from featureSelectionUtils import import_features
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler

class ModelPredict:

    def __init__(self, period, train_input_folder, fit, balance):
        self.period = period
        self.balance = balance
        self.folder_path = train_input_folder
        self.fit = fit
        self.main()


    def read_month(self):
        print('reading {} month data'.format(self.period))
        self.X = pd.read_csv('{}{}/selected_users_{}.tsv'.format(self.folder_path, self.period, self.period), sep='\t',dtype={"user_id":"string"})
        self.y = self.X['target'].copy()
        self.X.drop(['target','user_id'],axis=1, inplace=True)
       
        # select features
        features = import_features()
        self.X = self.X[features]
        
        # balance dataset
        if self.balance == True:
            undersample = RandomUnderSampler(sampling_strategy='majority')
            self.X , self.y = undersample.fit_resample(self.X, self.y)

    
    def read_month_select(self):
        return None


    def predict(self):
        if os.path.isfile('model.pkl'):
            model = load_model()
            scaler = load_scaler()
        else:
            print('Error: model does not exist')
            return 
        
        X_scaled = scaler.transform(self.X)
        y_pred = model.predict(X_scaled)
        score = f1_score(y_pred, self.y)
        print('Predict {} user suspension score: {}'.format(self.period, score))


    def main(self):
        if self.fit == True:
            self.read_month_select()
        else:
            self.read_month()
            self.predict()


