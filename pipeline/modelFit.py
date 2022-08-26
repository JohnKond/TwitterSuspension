import pandas as pd
from SaveLoadUtils import save_model, load_model, save_scaler, load_scaler
from featureSelectionUtils import import_features
from imblearn.under_sampling import RandomUnderSampler

class ModelFit:

    def __init__(self, period, train_folder, balance):
        self.period = period
        self.folder_path = train_folder
        self.balance = balance

        self.main()


    def read_month(self):
        self.X_train = pd.read_csv('{}{}/previous_users_{}.tsv'.format(self.folder_path, self.period, self.period),
                                   sep='\t', dtype={"user_id": "string"})
        self.y_train = self.X_train['target'].copy()
        self.X_train.drop(['target', 'user_id'], axis=1, inplace=True)

        # select features
        features = import_features()
        self.X_train = self.X_train[features]

        # balance dataset
        if self.balance == True:
            undersample = RandomUnderSampler(sampling_strategy='majority')
            self.X_train, self.y_train = undersample.fit_resample(self.X_train, self.y_train)

    def fit_model(self, X, y):

        # scale data
        X_scaled = self.scaler.transform(X.copy()) # transform or fit_transform
        self.model.fit(X_scaled, y)
        save_model(self.model)


    def main(self):
        self.read_month()
        self.fit_model(self.X_train, self.y_train)
        print('End fit model')
