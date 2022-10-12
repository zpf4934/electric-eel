# encoding: utf-8
"""
@author: andy
@file: electriceel.py
@time: 2022/1/9 下午3:26
@desc:
"""
import warnings
warnings.filterwarnings('ignore')
from ._feature_engine import FeatureEngine
from ._model_engine import ModelEngine

__version__ = '1.0.0'

class ElectricEel(object):
    def __init__(self, config):
        self.config = None
        self.MODEL_ENGINE = None
        self.FEATURE_ENGINE = None
        self.set_config(config)

    def set_config(self, config):
        self.config = config

    def feature_fit(self):
        """
        特征处理
        :return:
        """
        self.FEATURE_ENGINE = FeatureEngine(self.config)
        self.FEATURE_ENGINE.start_fit_engine()

    def feature_transform(self, df):
        """
        特征转换
        :param df: 原始特征
        :return:
        """
        assert self.FEATURE_ENGINE is not None, "特征处理未训练，请先进行feature_fit"
        df = self.FEATURE_ENGINE.start_transform_engine(df)
        return df

    def model_train(self):
        """
        模型训练
        :return:
        """
        self.MODEL_ENGINE = ModelEngine(self.config)
        self.MODEL_ENGINE.start_train_engine()

    def model_test(self, feature, metric = [], report = False, score = False, bins = 100):
        """
        模型测试
        :param feature: DataFrame 处理后的特征
        :param metric: list 评估指标列表，分类指标：['auc', 'ks', 'f1', 'precision_recall_curve']
        :param report: bool 是否产生报告
        :param score: bool 是否将概率转换为分数
        :param bins: int 分箱数
        :return:
        """
        assert self.MODEL_ENGINE is not None, "模型未训练，请先进行model_train"
        self.MODEL_ENGINE.start_test_engine(feature, metric, report, score, bins)

    def predict(self, feature):
        """
        预测
        :param feature: DataFrame 特征值
        :return:
        """
        assert self.MODEL_ENGINE is not None, "模型未训练，请先进行model_train"
        prod = self.MODEL_ENGINE.start_predict_engine(feature)
        return prod
