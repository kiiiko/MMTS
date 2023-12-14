import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
df = pd.read_excel('E:\zlx2\pythonProject1\data\guiji_data\侦察经度航迹规律提取\航迹4\data1.xlsx', engine='openpyxl')



# 提取经度和维度数据，使用列的索引号
longitudes = df.iloc[:, 0]  # 第1列
latitudes = df.iloc[:, 1]   # 第2列

# 获取经度和维度的最大值和最小值
lon_min, lon_max = longitudes.min(), longitudes.max()
lat_min, lat_max = latitudes.min(), latitudes.max()

# 使用matplotlib进行绘图
plt.figure(figsize=(10, 8))
plt.plot(longitudes, latitudes, label='Path', color='blue', marker='o', linestyle='-')
plt.title('Longitude vs Latitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xlim(lon_min-0.01, lon_max+0.01)  # 将范围设为最小值减1到最大值加1，为了使得图稍微有些边距
plt.ylim(lat_min-0.01, lat_max+0.01)
plt.grid(True)
plt.legend()
plt.show()

