# channel_modeling
This repo is for 2022 6G AI channel modeling.

## DDL:
2022/03/11（24:00:00）： 评测结束，榜单锁定

## 推荐宏包版本：
Python 3.7；

tensorflow 2.4.2(2.1.0+)；

pytorch > 1.7.0；

Numpy 1.18.1；

h5py 2.10.0

上传文件大小限制：文件大小不得超过400MB；

数据生成时间限制：总时间不得超过400秒。

## Note:
本赛题支持TensorFlow及Pytorch两种版本结果的提交，

大赛提供二者的模板，两类版本均提供generatorFun.py文件作为参考，其中：

1.generatorFun.py：包含分别针对两类信道的生成函数generator1及generator2，用于评测平台调用。
特别地，生成器函数的输入变量为生成数据的数量、生成器模型文件及真实信道文件；生成器函数输出为对应数量的生成信道；

2.evaluation.py：用于评测生成结果有效性，该函数将调用generatorFun.py中定义的生成器函数generator_1和generator_2并分别生成信道数据fake_1与fake_2，
用于与真实数据对比，评估生成数据的性能与分数（线上评测将使用独立的真实数据集合）