# encoding: utf-8
"""
@author: andy
@file: metric.py
@time: 2022/1/6 下午8:25
@desc:
"""
import numpy as np
import pandas as pd
from metric_platform.woe import WOE
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_recall_curve, mean_absolute_error, mean_squared_error, mean_squared_log_error

class Metric(object):
    def __init__(self, actual_label = None, predict = None):
        """
        评估指标
        :param actual_label: 真实标签
        :param predict: 预测标签
        """
        self.null_fre = None
        self.describe = None
        self.iv = None
        self.auc = None
        self.fpr = None
        self.tpr = None
        self.ks_thresholds = None
        self.ks = None
        self.f1 = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.pr_thresholds = None
        self.mean_absolute = None
        self.mean_squared = None
        self.mean_squared_log = None
        self.actual_label = actual_label
        self.predict = predict
        self.woe = WOE()
        self.to_array()

    def to_array(self):
        """
        转化数据格式，将数据转化为array
        :return:
        """
        if not isinstance(self.actual_label, np.ndarray):
            self.actual_label = np.array(self.actual_label)
        if not isinstance(self.predict, np.ndarray):
            self.predict = np.array(self.predict)

    def null_rate(self, df):
        """
        计算空值率
        :param df:
        :return:
        """
        self.null_fre = (df.isnull().sum() / len(df)).to_frame(name='missRate')
        self.null_fre = self.null_fre.sort_values('missRate', ascending=False)

    def iv_score(self, df, target):
        """
        计算特征iv值
        :param df: dataframe，原始数据
        :param target: str：标签列名
        :return:
        """
        df[target] = df[target].map(float)
        res_iv, x_col, woe_dict_list = self.woe.iv(X=df.drop([target], axis=1), y=df[target])
        self.iv = pd.concat([pd.DataFrame(x_col), pd.DataFrame(res_iv)], axis=1)
        self.iv.columns = ['feature', 'IV']
        self.iv = self.iv.sort_values('IV', ascending=False)
        return self.iv

    def desc(self, df):
        """
        特征数据描述
        :param df: dataframe：原始数据
        :return:
        """
        self.describe = df.describe().T

class ClassificationMetric(Metric):
    def __init__(self, actual_label, predict):
        super(ClassificationMetric, self).__init__(actual_label, predict)

    def auc_score(self):
        """
        计算auc
        :return:
        """
        self.auc = roc_auc_score(self.actual_label, self.predict)
        return self.auc

    def ks_score(self):
        """
        计算KS
        :return:
        """
        self.fpr, self.tpr, self.ks_thresholds = roc_curve(self.actual_label, self.predict)
        self.ks = max(self.tpr - self.fpr)
        return self.ks

    def f1_score(self):
        """
        计算F1
        :return:
        """
        precision, recall, pr_thresholds = self.precision_recall_curve()
        self.f1 = max(2 * (precision * recall) / (precision + recall))
        return self.f1

    def accuracy_score(self):
        """
        计算准确率，预测值必须为标签，暂时弃用
        :return:
        """
        self.accuracy = accuracy_score(self.actual_label, self.predict)
        return self.accuracy

    def precision_recall_curve(self):
        """
        准确率和召回率曲线
        :return:
        """
        self.precision, self.recall, self.pr_thresholds = precision_recall_curve(self.actual_label, self.predict)
        return self.precision, self.recall, self.pr_thresholds

class RegressionMetric(Metric):
    def __init__(self, actual_label, predict):
        super(RegressionMetric, self).__init__(actual_label, predict)

    def mean_absolute_error(self):
        self.mean_absolute = mean_absolute_error(self.actual_label, self.predict)
        return self.mean_absolute

    def mean_squared_error(self):
        self.mean_squared = mean_squared_error(self.actual_label, self.predict)
        return self.mean_squared

    def mean_squared_log_error(self):
        self.mean_squared_log = mean_squared_log_error(self.actual_label, self.predict)
        return self.mean_squared_log