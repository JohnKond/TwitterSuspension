from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb


def xgb_finetuning(X_train, y_train):
    print('XGBoost finetuning')
    model = xgb.XGBClassifier(
        # predictor='gpu_predictor',
        # tree_method='gpu_hist',
        objective='binary:logistic',
        use_label_encoder=False
    )
    param_grid = {
        'n_estimators': [300, 400, 500],
        'colsample_bytree': [0.65, 0.7, 0.75, 0.8, 0.9],
        'max_depth': [6, 7, 8, 9, 10, 11, 12],
        'gamma': [0, 0.25, 0.5, 1.0],
        'reg_alpha': [1.1, 1.2, 1.3],
        'reg_lambda': [0.1, 1, 1.1, 1.2, 1.3],
        'subsample': [0.6, 0.65, 0.7, 0.75, 0.8],
        'learning_rate': [0.005, 0.01, 0.015]
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
    learning_rate = params['learning_rate']
    gamma = params['gamma']

    clf = xgb.XGBClassifier(
        # predictor='gpu_predictor',
        # tree_method='gpu_hist',
        objective='binary:logistic',
        n_estimators=n_estimators,
        colsample_bytree=colsample_bytree,
        max_depth=max_depth,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        subsample=subsample,
        gamma=gamma,
        learning_rate=learning_rate,
        use_label_encoder=False,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)
    return clf
