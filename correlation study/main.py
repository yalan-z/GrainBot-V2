# !/user/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.inspection import plot_partial_dependence
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import r2_score


class XGBoostModel:
    def __init__(self, n_estimators=1000, max_depth=6, learning_rate=0.1, random_state=42,
                 reg_a=0, reg_l=0):
        self.params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': max_depth,
            'eta': learning_rate,
            'seed': random_state, 'silent': 1,
            'nthread': 10,
            'reg_alpha': reg_a,
            'reg_lambda': reg_l
        }
        self.n_estimators = n_estimators
        self.explainer = None

    def train(self, X_train, y_train):
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        self.model = xgb.train(self.params, dtrain, num_boost_round=self.n_estimators)

    def train_es(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=100):
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(data=X_val, label=y_val)
            evals.append((dval, 'validation'))
        self.model = xgb.train(self.params, dtrain, num_boost_round=self.n_estimators, evals=evals,
                               early_stopping_rounds=early_stopping_rounds)

    def evaluate(self, X_test, y_test):
        dtest = xgb.DMatrix(data=X_test, label=y_test)
        y_pred = self.model.predict(dtest)
        mask = y_test != 0
        y_test_nonzero = y_test[mask]
        y_pred_nonzero = y_pred[mask]
        rmspe = np.sqrt(np.mean(np.square((y_test_nonzero - y_pred_nonzero) / y_test_nonzero)))
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"Root Mean Squared Error: {rmse}")
        print(f"RMSPE: {rmspe}")
        print(f"R^2:{r2_score(y_test, y_pred)}")
        return rmse, rmspe

    def plot_pdp(self, X_train, feature_names, x_name='Feature', show=True, save_path=None):
        print("Generating Partial Dependence Plot...")
        model_sklearn = xgb.XGBRegressor(**self.params, n_estimators=self.n_estimators)
        model_sklearn.fit(X_train, self.model.predict(xgb.DMatrix(X_train)))

        plt.figure(figsize=(10, 6))
        plt.rcParams.update({'font.family': 'Arial', 'font.size': 16})
        display = PartialDependenceDisplay.from_estimator(
            model_sklearn, X_train, features=feature_names, grid_resolution=50
        )
        plt.xlabel(x_name, fontsize=18)
        plt.ylabel('Partial Dependence', fontsize=18)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        if show:
            plt.show()
        else:
            plt.close()
        return display

    def visual_shap(self, X_train, X_test, feature_names=None, show=True, save_path=None):
        print("Generating SHAP values...")
        self.explainer = shap.Explainer(self.model, X_train)
        shap_values = self.explainer(X_test)
        plt.figure(figsize=(8, 10))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        if show:
            plt.show()
        else:
            plt.close()
        return shap_values

    def plot_shap_dependence(self, shap_values, X_test, feature_index):
        shap.dependence_plot(feature_index, shap_values.values, X_test)

    def plot_ice(self, X_train, feature_names, kind='individual', x_name='Feature', show=True, save_path=None):
        print("Generating Partial Dependence Plot...")
        model_sklearn = xgb.XGBRegressor(**self.params, n_estimators=self.n_estimators)
        model_sklearn.fit(X_train, self.model.predict(xgb.DMatrix(X_train)))

        plt.figure(figsize=(10, 6))
        plt.rcParams.update({'font.family': 'Arial', 'font.size': 16})
        display = PartialDependenceDisplay.from_estimator(
            model_sklearn, X_train, features=feature_names, kind=kind, line_kw={'alpha': 0.5}
        )
        plt.xlabel(x_name, fontsize=18)
        plt.ylabel('Partial Dependence', fontsize=18)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        if show:
            plt.show()
        else:
            plt.close()
        return display

    def plot_scatter(self, X_test, y_test, show=True, save_path=None):

        dtest = xgb.DMatrix(data=X_test, label=y_test)
        y_pred = self.model.predict(dtest)

        plt.figure(figsize=(8, 6))
        plt.rcParams.update({'font.family': 'Arial', 'font.size': 16})
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

        plt.xlabel('True Values', fontsize=18)
        plt.ylabel('Predictions', fontsize=18)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        if show:
            plt.show()
        else:
            plt.close()


class LightGBMModel:
    def __init__(self, n_estimators=1000, max_depth=-1, learning_rate=0.1, random_state=42,
                 reg_alpha=0, reg_lambda=0):
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'n_jobs': -1
        }
        self.n_estimators = n_estimators
        self.model = None

    def train(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=10):
        import lightgbm as lgb

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val) if X_val is not None and y_val is not None else None

        callbacks = []
        if valid_data:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds,
                                                verbose=True))

        self.model = lgb.train(self.params, train_data, num_boost_round=self.n_estimators,
                               valid_sets=[train_data, valid_data] if valid_data else [train_data],
                               callbacks=callbacks)

    def evaluate(self, X_test, y_test):

        y_pred = self.model.predict(X_test)
        mask = y_test != 0
        y_test_nonzero = y_test[mask]
        y_pred_nonzero = y_pred[mask]
        rmspe = np.sqrt(np.mean(np.square((y_test_nonzero - y_pred_nonzero) / y_test_nonzero)))
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"Root Mean Squared Error: {rmse}")
        print(f"RMSPE: {rmspe}")
        return rmse, rmspe

    def plot_pdp(self, X_train, feature_names, x_name='Feature', show=True, save_path=None):
        from sklearn.inspection import PartialDependenceDisplay
        from lightgbm import LGBMRegressor
        import matplotlib.pyplot as plt

        print("Generating Partial Dependence Plot...")
        model_sklearn = LGBMRegressor(**self.params, n_estimators=self.n_estimators)
        model_sklearn.fit(X_train, self.model.predict(X_train))

        plt.figure(figsize=(10, 6))
        plt.rcParams.update({'font.family': 'Arial', 'font.size': 16})
        display = PartialDependenceDisplay.from_estimator(
            model_sklearn, X_train, features=feature_names, grid_resolution=50
        )
        plt.xlabel(x_name, fontsize=18)
        plt.ylabel('Partial Dependence', fontsize=18)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        if show:
            plt.show()
        else:
            plt.close()
        return display
