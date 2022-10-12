# encoding: utf-8
"""
@author: andy
@file: log.py
@time: 2022/2/9 下午5:26
@desc:
"""
import time
import logging

# 配置日志
def algo_logger(name):
    logger = logging.getLogger(name)
    fmt = '%(asctime)s %(pathname)s %(funcName)s %(lineno)s %(levelname)s - %(message)s'
    logging.basicConfig(format=fmt)
    logger.setLevel(logging.INFO)
    return logger

def log_info(logger, info_content):#info_content要显示的信息
    def decorator(func):#传入要执行的函数名称
        def wrapper(*args, **kwargs):
            start = time.time() * 1000
            logger.info("start {}".format(info_content))#这里是函数执行前要显示的信息
            ret = func(*args, **kwargs)#这里是真正要执行的函数
            logger.info('finish {}, use time {}ms'.format(info_content, int(time.time() * 1000 - start)))#这里是函数执行成功后显示的信息
            return ret
        return wrapper
    return decorator