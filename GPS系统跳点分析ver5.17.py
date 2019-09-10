# coding: utf-8

# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

# input filename
in1_f = "./output/2019-05-16-类型五：GPS系统跳点（待分析）.csv"
in2_f = "./output/2019-05-16-异常总结表.csv"

# output filename
out1_path = "./output/2019-05-16-类型五：GPS系统跳点.csv"
out2_path = "./output/2019-05-16-异常总结表.csv"

# 读取csv
in1_df = pd.read_csv(in1_f, header=None, encoding='gbk')
in2_df = pd.read_csv(in1_f, header=None, encoding='gbk')
# print(in1_df)

# 索引相减为新列
in1_df['index_diff'] = in1_df.groupby(11)[0].apply(lambda i: i.diff(1))
# print(in1_df)

# 重新命名列名
in1_df.columns = ['index', '_id', 'created_time', 'device_imei', 'direction', 'gps_count', 'latitude', 'longtitude',
                  'occurred_time', 'speed', 'start_timespan', 'vehicle_license_number', 'delta_T', 'delta_D',
                  'delta_v', 'datetime', 'type', 'index_diff']

# 筛选出确切的GPS系统跳点（特殊情况）
print("GPS系统跳点： ")
print(in1_df[(np.abs(in1_df['index_diff']) == 1)])
print()
result6_df = in1_df[(np.abs(in1_df['index_diff']) == 1)]
result6_df.to_csv(out1_path, mode='w', header=False, encoding='gbk')

# 异常点的summary表（添加第五种情况和更新后的总结表）
sum5_df = result6_df[['vehicle_license_number', 'datetime', 'type']]
sum5_df.to_csv(out2_path, mode='a', header=False, encoding='gbk')
