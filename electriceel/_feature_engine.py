# encoding: utf-8
"""
@author: andy
@file: _feature_engine.py
@time: 2022/2/8 上午9:36
@desc:
"""
from feature_platform import Fill, Scale, Bins, Encoder
from metric_platform import Metric
from ._base_engine import BaseEngine
from utils.log import algo_logger, log_info

logger = algo_logger(__file__)

class FeatureEngine(BaseEngine):
    """
    特征处理引擎，包含特征处理以及特征评估
    """
    def __init__(self, config):
        super(FeatureEngine, self).__init__(config)
        self.metric = Metric()
        self.source_num_col = None
        self.num_col = None
        self.source_cate_col = None
        self.cate_col = None
        self.fill = None
        self.scale = None
        self.bins = None
        self.encoder = None
        self.null_fre = None
        self.iv = None
        self.describe = None
        self.load_params()

    @log_info(logger, 'load params')
    def load_params(self):
        """
        参数加载
        :return:
        """
        assert "num_col" in self.config.keys() or "cate_col" in self.config.keys(), "未设置特征列"
        if "num_col" in self.config.keys():
            self.source_num_col = self.num_col = self.config.get('num_col')
        if "cate_col" in self.config.keys():
            self.source_cate_col = self.cate_col = self.config.get('cate_col')

    @log_info(logger, 'get feature null rate')
    def get_feature_null_rate(self):
        """
        特征缺失率
        :return: DataFrame
        """
        if 'null_rate' in self.config.get('metric').keys():
            self.metric.null_rate(self.feature)
            self.null_fre = self.metric.null_fre

    @log_info(logger, 'get feature iv')
    def get_feature_iv(self):
        """
        特征IV值计算
        :return: DataFrame
        """
        if 'iv' in self.config.get('metric').keys():
            assert self.target_col is not None, "未设置目标列"
            self.metric.iv_score(self.feature, self.target_col)
            self.iv = self.metric.iv

    @log_info(logger, 'get feature desc')
    def get_feature_desc(self):
        """
        计算特征值的描述信息
        :return: DataFrame
        """
        if 'describe' in self.config.get('metric').keys():
            self.metric.desc(self.feature)
            self.describe = self.metric.describe

    @log_info(logger, 'feature fill')
    def feature_fill(self):
        """
        缺失值填充
        :return:
        """
        if 'fill' in self.config.get('preprocessing').keys():
            self.fill = Fill(self.feature, self.num_col, self.cate_col, self.target_col)
            self.fill.fit(**self.config.get('preprocessing').get('fill'))
            self.feature = self.fill.data
            self.num_col = self.fill.num_col
            self.cate_col = self.fill.cate_col

    @log_info(logger, 'feature scale')
    def feature_scale(self):
        """
        特征标准化
        :return:
        """
        if 'scale' in self.config.get('preprocessing').keys():
            scale_col = self.config.get("preprocessing").get("scale").get("scale_col") if "scale_col" in self.config.get("preprocessing").get("scale").keys() else None
            self.scale = Scale(self.feature, self.num_col, self.cate_col, self.target_col, scale_col=scale_col)
            self.scale.fit(**self.config.get('preprocessing').get('scale'))
            self.feature = self.scale.data
            self.num_col = self.scale.num_col
            self.cate_col = self.scale.cate_col

    @log_info(logger, 'feature bins')
    def feature_bins(self):
        """
        特征分箱
        :return:
        """
        if 'bins' in self.config.get('preprocessing').keys():
            bins = self.config.get("preprocessing").get("bins").get("bin_col") if "bin_col" in self.config.get("preprocessing").get("bins").keys() else None
            self.bins = Bins(self.feature, self.num_col, self.cate_col, self.target_col, bin_col=bins)
            self.bins.fit(**self.config.get('preprocessing').get('bins'))
            self.feature = self.bins.data
            self.num_col = self.bins.num_col
            self.cate_col = self.bins.cate_col

    @log_info(logger, 'feature encoder')
    def feature_encoder(self):
        """
        特征编码
        :return:
        """
        if 'encoder' in self.config.get('preprocessing').keys():
            encoder = self.config.get("preprocessing").get("encoder").get("encoder_col") if "encoder_col" in self.config.get("preprocessing").get("encoder").keys() else None
            self.encoder = Encoder(self.feature, self.num_col, self.cate_col, self.target_col, encoder_col=encoder)
            self.encoder.fit(**self.config.get('preprocessing').get('encoder'))
            self.feature = self.encoder.data
            self.num_col = self.encoder.num_col
            self.cate_col = self.encoder.cate_col

    @log_info(logger, 'start fit engine')
    def start_fit_engine(self):
        """
        启动特征处理引擎
        :return:DataFrame 处理后的特征
        """
        pip_line = [('null_rate', self.get_feature_null_rate()), ('fill', self.feature_fill()), ('scale', self.feature_scale()),
                    ('bins', self.feature_bins()), ('encoder', self.feature_encoder()), ('iv', self.get_feature_iv()), ('describe', self.get_feature_desc())]
        for name, step in pip_line:
            step
        return self.feature

    @log_info(logger, 'start transform engine')
    def start_transform_engine(self, df):
        """
        特征转换
        :param df: 原始特征
        :return:
        """
        if self.fill:
            df = self.fill.transform(df)
        if self.scale:
            df = self.scale.transform(df)
        if self.bins:
            df, _ = self.bins.transform(df)
        if self.encoder:
            df, _ = self.encoder.transform(df)
        return df