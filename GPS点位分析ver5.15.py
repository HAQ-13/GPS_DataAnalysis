# coding: utf-8

# -*- coding:utf-8 -*
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from math import sin, asin, cos, radians, fabs, sqrt

np.set_printoptions(threshold=np.inf)
pd.set_option('display.width', 1000)  # 设置字符显示宽度
pd.set_option('display.max_rows', None)  # 设置显示最大行

# 数据预处理
# Json数据转为DataFrame数据
new_lines = []

filename = './export/2019-05-14-沪DB6898'

with open(filename) as f:
    for line in f:
        line = json.loads(line)

        for num in range(0, len(line["gps"])):
            new_line = {
                "_id": line["_id"]['$oid'],
                "device_imei": line["device_imei"],
                "start_timespan": line["start_timespan"]['$date'],
                "longtitude": line["gps"][num]['longtitude'],  # 此处按原数据写法
                "latitude": line["gps"][num]['latitude'],
                "direction": line["gps"][num]['direction'],
                "speed": line["gps"][num]['speed'],
                "created_time": line["gps"][num]['created_time']['$date'],
                "occurred_time": line["gps"][num]['occurred_time']['$date'],
                "gps_count": line["gps_count"],
                # "version": line["meta"]['version'],
                # "provider": line["meta"]['provider'],
                "vehicle_license_number": line["vehicle_license_number"]
            }
            new_lines.append(new_line)

df = pd.DataFrame(new_lines)

# 数据按时间（occurred_time）升序排列
df.sort_values('occurred_time', ascending=True, inplace=True)

# 检查时间（occurred_time）是否连续
# 相邻两行occurred_time计算差值，作为新列time_diff加入df表
df['time_diff'] = df.groupby('device_imei')['occurred_time'].apply(lambda i: i.diff(1))

# 统计差值time_diff中每个数值出现的频数
df['time_diff'].value_counts()

# 过滤出outlier所在行
# df[(np.abs(df['time_diff']) != 10000)]

# 查看数据格式
# df.dtypes

# 按需转换数据类型
df['latitude'] = df['latitude'].astype('double')
df['longtitude'] = df['longtitude'].astype('double')

# 计算两个GPS点位之间距离
EARTH_RADIUS = 6371  # 地球平均半径，6371km


def hav(theta):
    s = sin(theta / 2)
    return s * s


def get_distance_hav(lat1, lng1, lat2, lng2):
    # 经纬度转换成弧度
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    lng1 = radians(lng1)
    lng2 = radians(lng2)

    dlng = fabs(lng1 - lng2)
    dlat = fabs(lat1 - lat2)
    h = hav(dlat) + cos(lat1) * cos(lat2) * hav(dlng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))

    return distance


delta_d_line = []
# 插入ΔD一列第一个，数值为0
delta_d_line.insert(0, 0)

for i in range(1, len(df)):
    lng1, lat1 = (df.iloc[i, 6], df.iloc[i, 5])
    lng2, lat2 = (df.iloc[i - 1, 6], df.iloc[i - 1, 5])
    delta_d = get_distance_hav(lat1, lng1, lat2, lng2)
    # print(delta_d) # 单位：KM
    delta_d_line.append(delta_d)

delta_d_df = pd.DataFrame(delta_d_line)

# 添加ΔD一列至原有表df中
df = pd.concat([df, delta_d_df], axis=1)

# 重新命名列名
df.columns = ['_id', 'created_time', 'device_imei', 'direction', 'gps_count', 'latitude', 'longtitude', 'occurred_time',
              'speed', 'start_timespan', 'vehicle_license_number', 'delta_T', 'delta_D']

# ΔD的histogram图
hist1 = df['delta_D'].hist(bins='auto', grid=False, alpha=0.5)
plt.xlabel('ΔD')
plt.ylabel('Frequency')
plt.title('ΔD Distribution')
plt.show()

# ΔT的histogram图
hist2 = df['delta_T'].hist(bins='auto', grid=False, alpha=0.5)
plt.xlabel('ΔT')
plt.ylabel('Frequency')
plt.title('ΔT Distribution')
plt.show()

# 新列Δv=ΔD/ΔT
delta_v_line = []
delta_v_line.append(0)

for i in range(1, len(df)):
    if df.iloc[i, 11] != 0:
        delta_v_line.append((1000000 * df.iloc[i, 12]) / df.iloc[i, 11])
    else:
        delta_v_line.append(0)

delta_v_df = pd.DataFrame(delta_v_line)

# df添加新列Δv
df = pd.concat([df, delta_v_df], axis=1)

# 重新命名列名
df.columns = ['_id', 'created_time', 'device_imei', 'direction', 'gps_count', 'latitude', 'longtitude', 'occurred_time',
              'speed', 'start_timespan', 'vehicle_license_number', 'delta_T', 'delta_D',
              'delta_v']

# Δv的histogram图
hist3 = df['delta_v'].hist(bins='auto', grid=False, alpha=0.5)
plt.xlabel('Δv')
plt.ylabel('Frequency')
plt.title('Δv Distribution')
plt.show()

# 筛选出ΔT的outlier
print("ΔT的异常值： ")
print(df[(np.abs(df['delta_T']) != 10000)])
print()
# 特殊情况：ΔT为0时ΔD不为0
print("ΔT异常值的特殊情况（ΔT=0）： ")
print(df[(np.abs(df['delta_T']) == 0)])
print()
# 导出为csv文件
# result1_df.to_csv('ΔToutlier.csv',index=False,sep=',')


# 筛选出ΔD的outlier
print("ΔD的异常值： ")
print(df[(np.abs(df['delta_D']) > 0.25)])
print()

# 筛选出Δv的outlier
print("Δv的异常值： ")
print(df[(np.abs(df['delta_v']) > 30)])
print()

# 筛选出GPS缺失的点（ΔT和ΔD同时较大）
print("GPS数据缺失点（ΔT和ΔD同时较大）： ")
print(df[(np.abs(df['delta_D']) > 0.25) & (np.abs(df['delta_T']) != 10000)])
print()
