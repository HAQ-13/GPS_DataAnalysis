
# coding: utf-8

# In[ ]:


#-*- coding:utf-8 -*


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import codecs
from math import sin, asin, cos, radians, fabs, sqrt



# # 1 数据预处理

# ## 1.1 Json数据转化为table格式

# Table structure：
# device_imei   start_timespan   longtitude   latitude   occurred_time   vehicle_license_number

# In[2]:


# 读取json数据文件并转化为table

new_lines = []

filename =  'data_tel_沪DB6898'

with open(filename) as f:
    for line in f:
        line = json.loads(line)
        
        for num in range(0,len(line["gps"])):
            new_line = {
            "device_imei":line["device_imei"],
            "start_timespan":line["start_timespan"]['$date'],
            "longtitude":line["gps"][num]['longtitude'],
            "latitude":line["gps"][num]['latitude'],
            "occurred_time":line["gps"][num]['occurred_time']['$date'],
            "vehicle_license_number":line["vehicle_license_number"]
            }
            new_lines.append(new_line)
        
df = pd.DataFrame(new_lines)
df


# ## 1.2 数据按时间（occurred_time）升序排列

# group by device_imei，start_timespan

# In[3]:


df.sort_values(by = ['device_imei','start_timespan','occurred_time'],ascending = [True,True,True])


# ## 1.3 检查时间（occurred_time）是否连续

# 相邻两行occurred_time计算差值，作为新列加入df表

# In[4]:


df['diff'] = df.groupby('device_imei')['occurred_time'].apply(lambda i:i.diff(1))
df


# 统计差值中每个数值出现的频数

# In[5]:


df['diff'].value_counts()


# 过滤出outlier所在行

# In[6]:


df[(np.abs(df['diff']) > 10000)]


# # 2. 数据压缩

# 取定一个恰当的时间差Δt，每隔Δt选取一次数据，删去其他数据，组成新表1，降低数据处理量

# In[7]:


# 上表导出为csv文件
df.to_csv('C:/Users/64816/Desktop/HAQ INTERN/GPS点位判断/data_tel_沪DB6898.csv',encoding='utf-8')

# 每隔5行取1行（相当于每隔50s取一次GPS数据），存放到新csv文件中
with open('data_tel_沪DB6898.csv') as reader, open('new_data_tel_沪DB6898.csv', 'w') as writer:
    for index, line in enumerate(reader):
        if index % 5 == 0:
            writer.write(line)  


# # 3. 数据处理

# ## 3.1 相邻两点距离计算

# 根据新表1，根据经纬度距离计算公式，计算相邻两点之间的距离，结果为距离ΔD

# In[8]:


# 导入新表1

csv_f = codecs.open('new_data_tel_沪DB6898.csv','r',encoding='utf-8')
csv_df = pd.read_csv(csv_f)
csv_df


# In[10]:


# 计算两个GPS点位之间距离

#print(csv_df.iloc[0,1])

EARTH_RADIUS = 6371 # 地球平均半径，6371km
 
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

for i in range(1,len(csv_df)):
    lng1,lat1 = (csv_df.iloc[i,3], csv_df.iloc[i,2])
    lng2,lat2 = (csv_df.iloc[i-1,3], csv_df.iloc[i-1,2])
    delta_d = get_distance_hav(lat1,lng1,lat2,lng2)
    print(delta_d) # 单位：KM
    delta_d_line.append(delta_d)

delta_d_df = pd.DataFrame(delta_d_line)


# ## 3.2 组成新表2

# tj（t1开始自增）   tj+1（t2开始自增）   ΔD   Δt

# In[11]:


# 新表2的ΔD
delta_d_df


# In[12]:


# csv_df添加新列diff，计算压缩后的时间差Δt
csv_df['diff'] = csv_df.groupby('device_imei')['occurred_time'].apply(lambda i:i.diff(1))
csv_df


# In[13]:


# 取出新dataframe delta_t_df

delta_t_line = []

for i in range(1,len(csv_df)):
    print(csv_df.iloc[i,7])
    delta_t_line.append(csv_df.iloc[i,7])

delta_t_df = pd.DataFrame(delta_t_line)


# In[14]:


# 新表2的Δt
delta_t_df


# In[15]:


# 读取自增列csv文件

t_xlsx_file = "新表2自增列.xlsx"
t_xlsx_data = pd.read_excel(t_xlsx_file)
t_xlsx_df = pd.DataFrame(t_xlsx_data)

t_xlsx_df


# In[16]:


# 新表2
df_new2 = pd.concat([t_xlsx_df,delta_t_df,delta_d_df],axis=1)
df_new2


# In[19]:


# 新表2重命名列名
df_new2.columns = ['Tj','Tj+1','delta_t','delta_D']
df_new2


# # 4. 数据展现

# ## 4.1 histogram展现

# 设定Bin值，展现histogram图，观察outlier值（ΔD大于阈值）

# In[20]:


# 筛选出Δt时间内ΔD大于1的数据
df_new2[(np.abs(df_new2['delta_D']) > 1)]


# In[21]:


# 筛选出Δt大于设定时间差50s的数据
df_new2[(np.abs(df_new2['delta_t']) > 50000)]


# In[22]:


hist=df_new2['delta_D'].hist(bins=10,grid=False,color='#607c8e',alpha=0.5)
plt.xlabel('ΔD')
plt.ylabel('Frequency')
plt.title('ΔD Distribution')
plt.show()

