# encoding: utf-8
"""
@author: andy
@file: exception.py
@time: 2022/1/6 上午10:41
@desc:
"""
class ModelError(Exception):
    pass

class BacktestError(Exception):
    pass


if __name__ == '__main__':
    raise FileNotFoundError('文件错误')