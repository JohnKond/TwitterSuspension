from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from global_params import K_folds,XGB_params


def xgb_finetuning(X_train, y_train):
    print('XGBoost finetuning')
    model = xgb.XGBClassifier(
        predictor='gpu_predictor',
        tree_method='gpu_hist',
        num_classes=2,
        use_label_encoder=False
    )

    clf = GridSearchCV(
        estimator=model,
        param_grid=XGB_params,
        cv=StratifiedKFold(n_splits=K_folds, shuffle=True),
        n_jobs=-1,
        scoring='f1',
        verbose=2
    )

    clf.fit(X_train, y_train)

    print(clf.best_params_)
    print(clf.best_score_)
    return clf.best_params_, clf.best_score_


def xgb_run(clf, X_train, y_train):
    clf.fit(X_train, y_train)
    return clf
