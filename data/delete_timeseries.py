import os
import pandas as pd


def check_coordinates_validity(df):
    # 纬度的正常范围是 -90 到 90
    # 经度的正常范围是 -180 到 180
    latitude = df.iloc[:, 0]  # 第一列为纬度
    longitude = df.iloc[:, 1]  # 第二列为经度

    if not (-90 <= latitude.min() <= 90 and -90 <= latitude.max() <= 90):
        return False
    if not (-180 <= longitude.min() <= 180 and -180 <= longitude.max() <= 180):
        return False
    return True


directory = 'YOUR_DIRECTORY_PATH'  # 请替换为您的文件夹路径

for filename in os.listdir(directory):
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        file_path = os.path.join(directory, filename)
        df = pd.read_excel(file_path, header=None)  # 设置 header=None 以确保pandas不会自动将第一行作为列标签

        if not check_coordinates_validity(df):
            os.remove(file_path)
            print(f"Removed: {filename}")
