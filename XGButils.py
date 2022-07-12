import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


def xgb_finetuning(X_train, y_train):
    print('XGBoost finetuning')
    model = xgb.XGBClassifier(
        use_label_encoder=False
    )
    param_grid = {
        'n_estimators': [400, 700, 1000],
        'colsample_bytree': [0.7, 0.8],
        'max_depth': [15, 20, 25],
        'reg_alpha': [1.1, 1.2, 1.3],
        'reg_lambda': [1.1, 1.2, 1.3],
        'subsample': [0.7, 0.8, 0.9]
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True)      # 10 folds
    scoring = 'f1'

    clf = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring=scoring,
        verbose=2
    )

    clf.fit(X_train, y_train)
    results = clf.cv_results_
    print(clf.best_params_)
    print(clf.best_score_)
    return clf.best_params_, clf.best_score_


def xgb_run(params, X_train, y_train):
    n_estimators = params['n_estimators']
    colsample_bytree = params['colsample_bytree']
    max_depth = params['max_depth']
    reg_alpha = params['reg_alpha']
    reg_lambda = params['reg_lambda']
    subsample = params['subsample']

    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        colsample_bytree=colsample_bytree,
        max_depth=max_depth,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        subsample=subsample,
        use_label_encoder=False
    )

    clf.fit(X_train, y_train)
    return clf
