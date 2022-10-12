# encoding: utf-8
"""
@author: andy
@file: backtest.py
@time: 2022/1/7 下午2:55
@desc:
"""
import numpy as np
import pandas as pd

class Backtest(object):
    """
    用于生成回测报告
    """
    def __init__(self, col_label='label', bins=100):
        """
        Args:
            col_label: actual label string
            bins: num of bins
            splits: list of cutoff values for bins
            is_bad: Bool, ppmodel.predict_prob method
                    True: predicted bad/overdue probability
                    False: predicted good/not overdue probability
        """
        self.col_label = col_label
        self.bins = bins

    def get_splits(self, score):
        step = int(100 / self.bins)
        pct = list(range(0, 101, step))
        splits = np.percentile(score['prod'], pct)
        return splits

    def get_report(self, score):
        """
        正例：1，负例：0

        该报告主要针对业务人员，对模型整体效果进行说明，
        该报告会对整个测试集产生的测试结果（预测概率）进行区间划分，对各个区间段的区分情况进行统计，生成如下几个指标：
        - 拒绝率：低于该阈值的用户占比
        - 拒绝分数：该分段的阈值
        - 拒绝误判率：低于该阈值的用户中实际为正例的占比，（应该为1被判定为0）
        - 通过正判率：高于该阈值的用户中实际为正例的占比，（应该为1也被判定为1）
        - 分档通过率：在该分档区间（当前阈值和下一个阈值）实际是正例的占比
        - 分档个数：在该区间的用户量
        - 拒绝误判累计占比：低于该阈值的用户中预测为负例占实际正例的比例
        - 拒绝正判累计占比：低于该阈值的用户中预测为负例占实际负例的比例
        - 通过误判累计占比：高于该阈值的用户中预测为正例占实际负例的比例
        - 通过正判累计占比：高于该阈值的用户中预测为正例占实际正例的比例
        - K-S：拒绝正判累计占比 - 拒绝误判累计占比
        - 分档通过占比：在该分档区间实际正例的占比
        - 分档拒绝占比：在该分档区间实际负例的占比
        - 分档通过人数：在该分档区间实际正例的用户数
        - 分档拒绝人数：在该分档区间实际负例的用户数
        :param score:
        :return:
        """
        positive_count = score[self.col_label].sum()
        anti_count = len(score) - positive_count
        result = pd.DataFrame()
        self.splits = self.get_splits(score)

        splits_num = len(self.splits)
        for i in range(splits_num - 1):
            score_deny = self.splits[i]
            score_deny_plus = self.splits[i + 1]
            deny_df = score[score['prod'] <= score_deny_plus]
            accept_df = score[score['prod'] > score_deny_plus]

            if score_deny == score_deny_plus or deny_df.empty:
                continue

            deny_positive = deny_df[self.col_label].sum()
            deny_good = len(deny_df) - deny_positive
            deny_duerate = deny_positive / len(deny_df)

            accept_positive = accept_df[self.col_label].sum()
            accept_good = len(accept_df) - accept_positive

            accept_duerate = accept_df[self.col_label].sum() / len(accept_df)
            df_ix = score[(score['prod'] >= score_deny) & (score['prod'] < score_deny_plus)]
            counts = len(df_ix)
            bad_counts = df_ix[self.col_label].sum()
            good_counts = counts - bad_counts

            result.loc[i, '拒绝率'] = (i + 1) / (splits_num - 1)

            result.loc[i, '拒绝分数'] = score_deny_plus
            result.loc[i, '拒绝误判率'] = deny_duerate
            result.loc[i, '通过正判率'] = accept_duerate

            result.loc[i, '分档通过率'] = bad_counts / counts
            result.loc[i, '分档个数'] = counts

            result.loc[i, '拒绝误判累计占比'] = deny_positive / positive_count
            result.loc[i, '拒绝正判累计占比'] = deny_good / anti_count
            result.loc[i, '通过误判累计占比'] = accept_good / anti_count
            result.loc[i, '通过正判累计占比'] = accept_positive / positive_count
            result.loc[i, 'K-S'] = result.loc[i, '拒绝正判累计占比'] - result.loc[i, '拒绝误判累计占比']
            result.loc[i, '分档通过占比'] = bad_counts / counts
            result.loc[i, '分档拒绝占比'] = good_counts / counts
            result.loc[i, '分档通过人数'] = bad_counts
            result.loc[i, '分档拒绝人数'] = good_counts
        return result