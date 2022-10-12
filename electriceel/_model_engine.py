# encoding: utf-8
"""
@author: andy
@file: _model_engine.py
@time: 2022/2/8 下午1:42
@desc:
"""
import numpy as np
import pandas as pd

from ._base_engine import BaseEngine
from metric_platform import ClassificationMetric
from model import ClassificationModel
from utils import Backtest
from utils.log import algo_logger, log_info

logger = algo_logger(__file__)

class ModelEngine(BaseEngine):
    def __init__(self, config):
        super(ModelEngine, self).__init__(config)
        self.algorithm = None
        self.label = None
        self.params = None
        self.params2 = None
        self.all_cols = None
        self.model = None
        self.verbose = None
        self.prod = None
        self.auc = None
        self.ks = None
        self.f1 = None
        self.precision_recall_curve = None
        self.report = None
        self.feature_important = None
        self.load_params()

    @log_info(logger, 'load params')
    def load_params(self):
        assert "algorithm" in self.config.keys(), "未设置模型"
        self.algorithm = self.config.get('algorithm')
        assert self.target_col is not None, "未设置目标列"
        assert "params" in self.config.keys(), "未设置模型参数"
        self.params = self.config.get('params')
        self.params2 = self.config.get('params2') if 'params2' in self.config.keys() else {}
        self.verbose = self.config.get('verbose') if 'verbose' in self.config.keys() else True

    def set_final_col(self):
        self.all_cols = list(self.feature.columns)
        if self.target_col in self.all_cols:
            self.all_cols.remove(self.target_col)

    def set_label(self, feature):
        label = feature[[self.target_col]]
        return label

    def set_feature(self, feature):
        feature.drop([self.target_col], axis=1, inplace=True)
        return feature

    @log_info(logger, 'start train engine')
    def start_train_engine(self):
        """
        训练模型
        :return:
        """
        self.feature = self.format_col(self.feature)
        self.set_final_col()
        self.label = self.set_label(self.feature)
        self.feature = self.set_feature(self.feature)
        if self.model_type == 'classification':
            self.model = ClassificationModel(self.algorithm, self.feature, self.label)
            self.model.fit(self.params, self.params2, self.verbose)
            if 'feature_importance' in self.config.keys() and self.config.get('feature_importance'):
                self.model.feature_importance()
                self.feature_important = self.model.feature_important

    @log_info(logger, 'start test engine')
    def start_test_engine(self, feature, metric, report, score = False, bins = 100):
        """
        测试模型
        :param feature: DataFrame 处理后的特征
        :param metric: list 评估指标列表，分类指标：['auc', 'ks', 'f1', 'precision_recall_curve']
        :param report: bool 是否产生报告
        :param score: bool 是否将概率转换为分数
        :param bins: int 分箱数
        :return:
        """
        feature = self.format_col(feature)
        label = self.set_label(feature)
        feature = self.set_feature(feature)[self.all_cols]
        self.prod = self.model.predict(feature)
        if metric:
            self.get_metric(self.label, self.prod[:, 1], metric)

        if report:
            label['prod'] = self.prod[:, 1]
            self.get_report(label, bins, score)

    @log_info(logger, 'start predict engine')
    def start_predict_engine(self, feature):
        """
        预测
        :param feature: DataFrame 处理后的特征
        :return:
        """
        feature = self.format_col(feature)
        feature = feature[self.all_cols]
        prod = self.model.predict(feature)
        return prod

    def get_metric(self, label, prod, metric=None):
        """
        获取评估结果
        :param label: DataFrame 原始标签
        :param prod: DataFrame 预测标签
        :param metric: list 评估指标
        :return:
        """
        if self.model_type == 'classification':
            metric = ['auc', 'ks', 'f1', 'precision_recall_curve'] if metric is None else metric
            metric_engine = ClassificationMetric(label, prod)
            for m in metric:
                if m == 'auc':
                    self.auc = metric_engine.auc_score()
                if m == 'ks':
                    self.ks = metric_engine.ks_score()
                if m == 'f1':
                    self.f1 = metric_engine.f1_score()
                if m == 'precision_recall_curve':
                    precision, recall, pr_thresholds = metric_engine.precision_recall_curve()
                    pr_thresholds = np.append(pr_thresholds, max(prod[:, 1]) + 1)
                    self.precision_recall_curve = pd.DataFrame(np.array([precision, recall, pr_thresholds]).T, columns=['precision', 'recall', 'pr_thresholds'])

    def get_report(self, label, bins = None, score = False):
        """
        获取回测报告
        :param label: DataFrame columns=target_col,prod
        :param bins: int 分箱数
        :param score: bool 是否转换为分数
        :return:
        """
        bins = bins if bins else 100
        bt = Backtest(self.target_col, bins)
        if score:
            score = self.tool.prob_to_score(self.label['prod'])
            self.label['prod'] = score['score']
        self.report = bt.get_report(label)