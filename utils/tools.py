# encoding: utf-8
"""
@author: andy
@file: tools.py
@time: 2022/1/6 上午9:52
@desc:
"""
import pickle
import pandas as pd
import numpy as np

def load_data(file_path : str, *args, **kwargs):
    """
    加载本地文件，支持csv,pkl,h5
    :param file_path: 文件地址
    :param args:
    :param kwargs:
    :return:
    """
    if file_path.endswith('csv'):
        source_data = pd.read_csv(file_path, *args, **kwargs)
    elif file_path.endswith('pkl'):
        source_data = pd.read_pickle(file_path)
    elif file_path.endswith('h5'):
        source_data = pd.read_hdf(file_path)
    else:
        raise ValueError('不支持的文件格式')
    return source_data

def prob_to_score(prob, A=441.5, B=-72.135, max_score=900, min_score=300):
    """
    从模型概率生成分值
    Args:
        prob: probability
    """
    logit = np.log(prob / (1 - prob))
    score = A + B * logit
    result = pd.DataFrame()
    result['score'] = score
    result['score'].where(result['score'] <= max_score, max_score, inplace=True)
    result['score'].where(result['score'] >= min_score, min_score, inplace=True)
    return result

def save_model(model, file_path):
    with open(file_path, 'wb') as fw:
        pickle.dump(model, fw)

def load_model(file_path):
    with open(file_path, 'rb') as fr:
        model = pickle.load(fr)
        return model

def format_columns(feature):
    temp = {}
    for column in feature:
        temp[column] = column.replace('[', '').replace(']', '').replace(',', '')
    return temp