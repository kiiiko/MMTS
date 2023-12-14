import os
import pandas as pd


def extract_coordinates(folder_path):
    """
    从给定文件夹的Excel文件中提取经度和纬度。

    参数:
    - folder_path: 要读取Excel文件的文件夹路径

    返回:
    - coords_list: 一个列表，其中每个元素是一个元组，包含该文件的经度和纬度列表
    """

    # 获取文件夹内所有的Excel文件
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') or f.endswith('.xls')]

    coords_list = []

    for file in excel_files:
        file_path = os.path.join(folder_path, file)

        # 读取Excel文件
        df = pd.read_excel(file_path)

        # 假设经度在第一列，纬度在第二列
        longitudes = df.iloc[:, 2].tolist()
        latitudes = df.iloc[:, 3].tolist()

        coords_list.append((longitudes, latitudes))

    return coords_list



