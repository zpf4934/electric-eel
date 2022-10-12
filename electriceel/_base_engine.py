# encoding: utf-8
"""
@author: andy
@file: _base_engine.py
@time: 2022/2/8 下午1:52
@desc:
"""
from utils import load_data, format_columns

class BaseEngine(object):
    def __init__(self, config):
        self.config = config
        self.model_type = None
        self.feature = None
        self.target_col = None
        self.base_params()

    def base_params(self):
        assert "feature" in self.config.keys(), "未设置数据集"
        if isinstance(self.config.get('feature'), str):
            self.feature = load_data(self.config.get('feature'))
        else:
            self.feature = self.config.get('feature')
        self.target_col = self.config.get('target_col') if "target_col" in self.config.keys() else None
        assert "model_type" in self.config.keys(), "未设置模型类型"
        self.model_type = self.config.get('model_type')

    def format_col(self, feature):
        columns = format_columns(feature.columns.tolist())
        feature.rename(columns=columns, inplace=True)
        return feature