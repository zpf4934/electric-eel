# encoding: utf-8
"""
@author: andy
@file: classification.py
@time: 2022/1/7 上午11:13
@desc:
"""
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

from utils import ModelError


class ClassificationBase(object):
    def __init__(self, model_type, feature, label):
        self.model_type = model_type
        self.feature = feature
        self.label = label
        self.feature_important = None
        self.reports = {}

    def feature_importance(self):
        if not isinstance(self.feature, pd.DataFrame):
            raise ModelError('Input must be DataFrame')
        self.feature_important = pd.DataFrame()
        self.feature_important['变量值'] = [x for x in self.feature.columns]
        if self.model_type in ['lr', 'svm']:  # 如果用逻辑回归，则model.coef_[0]为特征重要性
            self.feature_important['分值'] = [round(-x * 100, 2) for x in self.model.coef_[0]]  # model.coef_[0]模型运算结果round(-x*100, 2)，对-x*100计算结果保留2位小数
        elif self.model_type in ['xgb', 'rf', 'gbdt', 'lgb', 'gbdt+lr', 'rf+lr', 'xgb+lr', 'lgb+lr']:
            self.feature_important['分值'] = [round(x, 4) for x in self.model.feature_importances_]  # 如果不是逻辑回归，则model.feature_importances_为特征重要性


class ClassificationModel(ClassificationBase):
    def __init__(self, model_type, feature, label):
        super(ClassificationModel, self).__init__(model_type, feature, label)
        self.model = None
        self.one_hot_encoder = None
        self.next_model = None

    def fit(self, params, params2=None, verbose=True):  # 给xgboost喂入数据，在主函数中调用fit，把df_train->data
        """
        :param params: first model param dict
        :param params2: seconde model param dict
        :param verbose: show log
        :return: model
        """
        if self.model_type == 'xgb':
            self.model = xgb.XGBClassifier(**params)
            eval_set = [(self.feature, self.label.values.reshape(self.label.values.size, ))]
            self.model.fit(self.feature, self.label.values.reshape(self.label.values.size, ), eval_metric=['auc','logloss'], eval_set=eval_set, verbose=verbose,
                           early_stopping_rounds=5)  # 给模型喂入训练数据
        elif self.model_type == 'lgb':
            self.model = lgb.LGBMClassifier(**params)
            eval_set = [(self.feature, self.label.values.reshape(self.label.values.size, ))]
            self.model.fit(self.feature, self.label.values.reshape(self.label.values.size, ), eval_set=eval_set, eval_metric=['auc','logloss'])
        elif self.model_type == 'lr':
            self.model = LogisticRegression(**params)
            self.model.fit(self.feature, self.label.values.reshape(self.label.values.size, ))
        elif self.model_type == 'svm':
            self.model = SVC(probability=True, **params)
            self.model.fit(self.feature, self.label.values.reshape(self.label.values.size, ))
        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(**params)
            self.model.fit(self.feature, self.label.values.reshape(self.label.values.size, ))
        elif self.model_type == 'gbdt':
            self.model = GradientBoostingClassifier(**params)
            self.model.fit(self.feature, self.label.values.reshape(self.label.values.size, ))
        elif self.model_type == 'gbdt+lr':
            self.gbdt_lr(params, params2)
        elif self.model_type == 'rf+lr':
            self.rf_lr(params, params2)
        elif self.model_type == 'xgb+lr':
            self.xgb_lr(params, params2, verbose)
        elif self.model_type == 'lgb+lr':
            self.lgb_lr(params, params2)

    def gbdt_lr(self, params, params2):
        self.model = GradientBoostingClassifier(**params)
        self.model.fit(self.feature, self.label.values.reshape(self.label.values.size, ))
        new_feature = self.model.apply(self.feature).reshape(-1, self.model.n_estimators)
        self.integration_lr(new_feature, params2)

    def rf_lr(self, params, params2):
        self.model = RandomForestClassifier(**params)
        self.model.fit(self.feature, self.label.values.reshape(self.label.values.size, ))
        new_feature = self.model.apply(self.feature).reshape(-1, self.model.n_estimators)
        self.integration_lr(new_feature, params2)

    def xgb_lr(self, params, params2, verbose):
        self.model = xgb.XGBClassifier(**params)
        eval_set = [(self.feature, self.label.values.reshape(self.label.values.size, ))]
        self.model.fit(self.feature, self.label.values.reshape(self.label.values.size, ), eval_metric=['auc','logloss'], eval_set=eval_set, verbose=verbose,
                       early_stopping_rounds=5)
        new_feature = self.model.apply(self.feature).reshape(-1, self.model.best_ntree_limit)
        self.integration_lr(new_feature, params2)

    def lgb_lr(self, params, params2):
        self.model = lgb.LGBMClassifier(**params)
        eval_set = [(self.feature, self.label.values.reshape(self.label.values.size, ))]
        self.model.fit(self.feature, self.label.values.reshape(self.label.values.size, ), eval_set=eval_set, eval_metric=['auc','logloss'])
        new_feature = self.model.predict(self.feature, pred_leaf=True).reshape(-1, self.model.n_estimators)
        self.integration_lr(new_feature, params2)

    def integration_lr(self, new_feature, params2):
        self.one_hot_encoder = OneHotEncoder()
        self.one_hot_encoder.fit(new_feature)
        new_feature = self.one_hot_encoder.transform(new_feature)
        self.next_model = LogisticRegression(**params2)
        self.next_model.fit(new_feature, self.label.values.reshape(self.label.values.size, ))

    def n_estimators_transform(self, df):
        if self.model_type == 'lgb+lr':
            feature = self.model.predict(df, pred_leaf=True).reshape(-1, self.model.n_estimators)
        else:
            feature = self.model.apply(df).reshape(-1, self.model.n_estimators)
        feature = self.one_hot_encoder.transform(feature)
        prod = self.next_model.predict_proba(feature)
        return prod

    def best_tree_limit_transform(self, df):
        feature = self.model.apply(df).reshape(-1, self.model.best_ntree_limit)
        feature = self.one_hot_encoder.transform(feature)
        prod = self.next_model.predict_proba(feature)
        return prod

    def combination_model(self, df):
        prod = None
        assert self.model is not None and self.next_model is not None, '模型未训练'
        if self.model_type == 'gbdt+lr':
            prod = self.n_estimators_transform(df)
        elif self.model_type == 'rf+lr':
            prod = self.n_estimators_transform(df)
        elif self.model_type == 'xgb+lr':
            prod = self.best_tree_limit_transform(df)
        elif self.model_type == 'lgb+lr':
            prod = self.n_estimators_transform(df)
        return prod

    def predict(self, df):
        prod = None
        assert self.model is not None, '模型未训练'
        if self.model_type == 'xgb':
            prod = self.model.predict_proba(df)
        elif self.model_type == 'lr':
            prod = self.model.predict_proba(df)
        elif self.model_type == 'svm':
            prod = self.model.predict_proba(df)
        elif self.model_type == 'rf':
            prod = self.model.predict_proba(df)
        elif self.model_type == 'gbdt':
            prod = self.model.predict_proba(df)
        elif self.model_type == 'gbdt+lr':
            prod = self.combination_model(df)
        elif self.model_type == 'rf+lr':
            prod = self.combination_model(df)
        elif self.model_type == 'xgb+lr':
            prod = self.combination_model(df)
        elif self.model_type == 'lgb+lr':
            prod = self.combination_model(df)
        return prod
