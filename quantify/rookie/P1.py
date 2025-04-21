
"""
    金融时间序列分析
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')


# TODO: 导入数据
# 1.时间为索引
# 2.将时间转换成标准格式
# 3.绘制各个指标的走势情况

# data1 = pd.read_csv('./stock_data/AAPL.csv', index_col=0, parse_dates=True)
# print(data1)
# data1.plot(figsize=(10, 12), subplots=True)
# plt.show()


# TODO: 统计分析
# 1.数据中各项指标统计结果
# 2.使用aggregate方法将多种统计指标汇总
# data1 = pd.read_csv('./stock_data/AAPL.csv', index_col=0, parse_dates=True)
# # data1.info()
# # print(data1.describe().round(2))
# print(data1.aggregate([min, max, np.mean, np.std, np.median]))



# TODO： 序列变化情况计算




# TODO：时间序列重采样


