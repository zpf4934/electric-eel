## 模型算法平台
### 文件结构

#### data:数据存储

- feature_platform:特征处理
- Fill：缺失值填充
- Scale：特征标准化
- Bins：分箱
- Encoder：特征编码
#### metric_platform:评估模块
- null_rate：缺失率
- iv_score：iv值
- desc：特征描述

1. ClassificationMetric：分类模型评估
- auc_score
- ks_score
- f1_score
- precision_recall_curve：精准率和召回率曲线

2. RegressionMetric：回归模型评估
- mean_absolute_error
- mean_squared_error
- mean_squared_log_error
#### model:模型模块

1. ClassificationModel：分类模型
- lr
- xgb
- svm
- rf
- gbdt
- gbdt+lr
- rf+lr
- xgb+lr
- lgb+lr
#### util：工具模块
- load_data：加载文件数据
- prob_to_score：概率转换为分数
- save_model：模型保存
- load_model：模型加载
### 使用说明
根据electric_eel.py生成ElectricEel对象
- 特征处理
配置feature_config，调用feature_fit()函数进行特征处理，返回处理后的特征及相关信息
- 模型训练
配置train_config，调用model_train()函数，进行模型训练，该步骤的特征数据可依赖上一步特征处理后的数据，也可以自定义数据。如果配置特征重要性，会返回特征重要性
- 模型测试
调用model_test()函数，进行模型测试，该步骤依赖于上一步产生的模型，及第一步的特征处理，返回预测值以及相应的评测结果
- 模型预测
输入原始特征，依赖于第一步产生的特征处理及第二步模型训练，进行预测
- 特征输出
如果仅需要进行特征处理，则在第一步完成后，即可，调用feature_transform()输入原始特征即可输出处理后的特征  

### 配置说明
**feature_config**

```
{
    "feature": "dataframe或者filepath：训练数据,必填",
    "num_col": "list：数值型特征名列表,必填",
    "cate_col": "list：类别型特征名列表,必填",
    "target_col": "str：目标列名,选填",
    "model_type":"str：classification:分类，regression:回归,必填",
    "metric": {特征评估，必填
        "null_rate":"bool:是否计算缺失率,选填",
        "iv":"bool:是否计算IV值,选填",
        "describe":"bool:是否获取特征描述信息,选填"
    },
    "preprocessing": {特征处理，必填
        "fill": {数据填充，选填
            "method": "str:填充方式，simple：简单采样,KNN：knn算法采样填充，default:simple，选填",
            "fill_cate": "bool 类别型特征是否填充，default:True，选填",
            "fill_value": "dict 指定特征列的填充值，default:None，选填",
            "num_sample_strategy": "str  数值型填充方式，mean：平均值，median：中位数，most_frequent：频率，default:median，选填",
            "cate_sample_strategy": "str 类别型填充方式，most_frequent：频率，default:most_frequent，选填",
            "n_neighbors": "int knn采样邻居个数，default：5，选填"
        },
        "scale": {标准化，选填
            "method": "str 选择处理方式，max_abs_scaler，min_max_scaler，standard_scaler，default:standard_scaler，选填",
            "scale_col": "list 需要标准化的特征列，default：数值型特征列:，选填"
        },
        "bins": {分箱，选填
            "bin_col": "list 需要分箱的特征列，default：数值型特征列:，选填",
            "n_bins": "int or array-like of shape (n_features,), default=5，选填",
            "encode": "用来编码转换结果的方法。{‘onehot’, ‘onehot-dense’, ‘ordinal’}, default=’onehot’,选填
                        'onehot'。用one-hot编码对转换后的结果进行编码，并返回一个稀疏矩阵。忽略的特征总是向右叠加。
                        'onehot-dense'。对转换后的结果进行单热编码，并返回一个密集数组。忽略的特征总是堆积在右边。
                        'ordinal'。返回编码为整数的bin标识符。",
            "strategy": "用来定义分仓宽度的策略。{‘uniform’, ‘quantile’, ‘kmeans’}, default=’quantile’，选填
                        'uniform'。每个特征中的所有箱体都有相同的宽度。
                        ‘quantile’。每个特征中的所有分仓都有相同的点数。
                        'kmeans'。每个bin中的值都有相同的一维k-means集群的最近中心。"
        },
        "encoder": {编码，选填
            "encoder_col": "list 需要编码的特征列，default：类别型特征列，选填",
            "method": "str:one-hot or ordinal，default：one-hot:，选填"
        }
    }
}
```
**train_config**
```
{
    "algorithm": "str:模型，{'xgb','lr','svm','rf','gbdt','gbdt+lr','rf+lr','xgb+lr'，'lgb+lr'},必填",
    "feature": "dataframe：特征数据,必填",
    "target_col": "str：目标列名,选填",
    "model_type":"str：classification:分类，regression:回归,必填",
    "params": "dick：第一模型参数,必填",
    "params2":"dick：组合模型中第二模型参数,必填",
    "verbose": "bool:是否显示日志，default:True,选填",
    "feature_importance": "bool:是否计算特征重要性，default:True,选填"
}
```

### 回归测试报告
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

**用例参考example.py文件**
