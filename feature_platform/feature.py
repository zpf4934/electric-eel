# encoding: utf-8
"""
@author: andy
@file: feature.py
@time: 2022/1/6 上午11:29
@desc:
"""
from sklearn import impute
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, KBinsDiscretizer, OneHotEncoder, OrdinalEncoder
import pandas as pd
from scipy import sparse
from utils import load_data

class Feature(object):
    def __init__(self, data, num_col, cate_col, target_col):
        self.num_col = num_col
        self.cate_col = cate_col
        self.target_col = target_col
        self.data = data
        self.load_data()
        self.unity_cols()
        # self.unity_type()

    def fit(self):
        pass

    def transform(self, data : pd.DataFrame):
        pass

    def load_data(self):
        if isinstance(self.data, str):
            self.data = load_data(self.data)

    def unity_cols(self):
        if self.target_col:
            self.data = self.data[self.num_col + self.cate_col + [self.target_col]]
        else:
            self.data = self.data[self.num_col + self.cate_col]

    def unity_type(self):
        self.data[self.num_col] = self.data[self.num_col].applymap(float)
        self.data[self.cate_col] = self.data[self.cate_col].applymap(str)

class Fill(Feature):
    def __init__(self, data, num_col, cate_col, target_col):
        super(Fill, self).__init__(data, num_col, cate_col, target_col)
        self.fill_value = None
        self.num_impute = None
        self.cate_impute = None

    def fit(self, method = 'simple', fill_cate = True, fill_value = None, num_sample_strategy = 'median', cate_sample_strategy = 'most_frequent', n_neighbors = 5):
        """
        缺失值填充
        :param n_neighbors: int knn采样邻居个数default：5
        :param cate_sample_strategy: str 类别型填充方式，most_frequent：频率
        :param num_sample_strategy: str  数值型填充方式，mean：平均值，median：中位数，most_frequent：频率
        :param method:str 填充方式，可选sample：简单采样,KNN：knn算法采样填充，default:sample
        :param fill_cate:bool 类别型特征是否填充
        :param fill_value: dict 指定特征列的填充值
        :return:
        """
        self.fill_value = fill_value
        if fill_value:
            self.data.fillna(fill_value, inplace=True)
        if method == 'simple':
            self.num_impute = impute.SimpleImputer(strategy=num_sample_strategy)
        elif method == 'KNN':
            self.num_impute = impute.KNNImputer(n_neighbors=n_neighbors)

        if fill_cate:
            self.cate_impute = impute.SimpleImputer(strategy=cate_sample_strategy)

        if self.num_impute:
            self.num_impute.fit(self.data[self.num_col])
            fill = self.num_impute.transform(self.data[self.num_col])
            self.data.update(pd.DataFrame(fill, columns=self.num_col))
        if self.cate_impute:
            self.cate_impute.fit(self.data[self.cate_col])
            fill = self.cate_impute.transform(self.data[self.cate_col])
            self.data.update(pd.DataFrame(fill, columns=self.cate_col))

    def transform(self, data : pd.DataFrame):
        if self.fill_value:
            data.fillna(self.fill_value, inplace=True)
        if self.num_impute:
            fill = self.num_impute.transform(data[self.num_col])
            data.update(pd.DataFrame(fill, columns=self.num_col))
        if self.cate_impute:
            fill = self.cate_impute.transform(data[self.cate_col])
            data.update(pd.DataFrame(fill, columns=self.cate_col))
        return data

class Scale(Feature):
    def __init__(self, data, num_col, cate_col, target_col, scale_col = None):
        super(Scale, self).__init__(data, num_col, cate_col, target_col)
        self.scale_model = None
        self.scale_col = scale_col if scale_col else num_col

    def fit(self, method = 'standard_scaler'):
        """
        对特征进行标准化或归一化处理
        :param method: str 选择处理方式，max_abs_scaler，min_max_scaler，standard_scaler
        :return:
        """
        if method == 'max_abs_scaler':
            self.scale_model = MaxAbsScaler()
        elif method == 'min_max_scaler':
            self.scale_model = MinMaxScaler()
        elif method == 'standard_scaler':
            self.scale_model = StandardScaler()
        else:
            raise ValueError("请指定正确的方法")
        self.scale_model.fit(self.data[self.scale_col])
        self.data = self.transform(self.data)

    def transform(self, data):
        assert self.scale_model is not None, '模型未训练'
        scale_data = self.scale_model.transform(data[self.scale_col])
        data.update(pd.DataFrame(scale_data, columns=self.scale_col))
        return data

class Bins(Feature):
    def __init__(self, data, num_col, cate_col, target_col, bin_col = None):
        super(Bins, self).__init__(data, num_col, cate_col, target_col)
        self.bin_col = bin_col if bin_col else num_col
        self.bin_model = None
        self.encode = None

    def fit(self, n_bins=5, encode='ordinal', strategy='uniform'):
        """
        对特征进行分箱
        :param n_bins: int or array-like of shape (n_features,), default=5，要产生的分箱的数量
        :param encode: 用来编码转换结果的方法。{‘onehot’, ‘onehot-dense’, ‘ordinal’}, default=’onehot’
                        'onehot'。用one-hot编码对转换后的结果进行编码，并返回一个稀疏矩阵。忽略的特征总是向右叠加。
                        'onehot-dense'。对转换后的结果进行单热编码，并返回一个密集数组。忽略的特征总是堆积在右边。
                        'ordinal'。返回编码为整数的bin标识符。
        :param strategy:用来定义分仓宽度的策略。{‘uniform’, ‘quantile’, ‘kmeans’}, default=’quantile’
                        'uniform'。每个特征中的所有箱体都有相同的宽度。
                        ‘quantile’。每个特征中的所有分仓都有相同的点数。
                        'kmeans'。每个bin中的值都有相同的一维k-means集群的最近中心。
        :return:
        """
        self.encode = encode
        self.bin_model = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
        self.bin_model.fit(self.data[self.bin_col])
        self.data, self.num_col = self.transform(self.data)

    def transform(self, data):
        assert self.bin_model is not None, '模型未训练'
        bin_data = self.bin_model.transform(data[self.bin_col])
        if isinstance(bin_data, sparse.csr_matrix):
            bin_data = bin_data.toarray()
        data.drop(self.bin_col, axis=1, inplace=True)
        columns = self.bin_model.feature_names_in_ if self.encode == 'ordinal' else self.bin_model.get_feature_names_out()
        data = pd.concat([data, pd.DataFrame(bin_data, columns=columns)], axis=1)
        num_col = sorted(list(set(self.num_col) - set(self.bin_col) | set(columns)))
        return data, num_col

class Encoder(Feature):
    def __init__(self, data, num_col, cate_col, target_col, encoder_col = None):
        super(Encoder, self).__init__(data, num_col, cate_col, target_col)
        self.encoder_col = encoder_col if encoder_col else cate_col
        self.encoder_model = None
        self.method = None
        self.data[self.encoder_col] = self.data[self.encoder_col].applymap(str)

    def fit(self, method = 'one-hot'):
        assert self.encoder_col, "未设置编码列"
        if method == 'one-hot':
            self.encoder_model = OneHotEncoder(handle_unknown='ignore')
        elif method == 'ordinal':
            self.encoder_model = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value = -1)
        else:
            raise ValueError("请指定正确的方法")
        self.method = method
        self.encoder_model.fit(self.data[self.encoder_col])
        self.data, self.cate_col = self.transform(self.data)

    def transform(self, data):
        assert self.encoder_model is not None, '模型未训练'
        encoder_data = self.encoder_model.transform(data[self.encoder_col])
        if isinstance(encoder_data, sparse.csr_matrix):
            encoder_data = encoder_data.toarray()
        columns = []
        if self.method == 'one-hot':
            for i in range(len(self.encoder_model.feature_names_in_)):
                for cate in self.encoder_model.categories_[i]:
                    columns.append('{}_{}'.format(self.encoder_model.feature_names_in_[i], cate))
        else:
            columns = self.encoder_model.feature_names_in_
        data.drop(self.encoder_col, axis=1, inplace=True)
        data = pd.concat([data, pd.DataFrame(encoder_data, columns=columns)], axis=1)
        cate_col = sorted(list(set(self.cate_col) - set(self.encoder_col) | set(columns)))
        return data, cate_col