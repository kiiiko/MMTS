import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

def load_data_from_subfolder(subfolder_path, lat_column, lon_column):
    """从子文件夹中加载数据"""
    data_subfolder = []

    for file in os.listdir(subfolder_path):
        if file.endswith('.xlsx'):
            file_path = os.path.join(subfolder_path, file)

            try:
                df = pd.read_excel(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            # 检查经纬度的范围
            if ((df.iloc[:, lon_column] < -180) | (df.iloc[:, lon_column] > 180) |
                (df.iloc[:, lat_column] < -90) | (df.iloc[:, lat_column] > 90)).any():
                print(f"Removing {file_path} due to invalid data.")
                os.remove(file_path)
                continue

            coordinates = df.iloc[:, [lat_column, lon_column]].values.tolist()
            data_subfolder.append(coordinates)  # 整个Excel文件为一个数据点

    return data_subfolder

def load_data_from_directory(base_path):
    """从基础路径中的子文件夹加载数据"""
    all_data = []
    all_labels = []
    subfolders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    for label, subfolder in enumerate(subfolders):
        folder_path = os.path.join(base_path, subfolder)
        data_folder = load_data_from_subfolder(folder_path, 0, 1)
        labels_folder = [label] * len(data_folder)

        all_data.extend(data_folder)
        all_labels.extend(labels_folder)

    return all_data, all_labels

# 主程序开始
base_path = "E:/zlx2/pythonProject1/data/guiji_data/侦察经度航迹规律提取"
data, labels = load_data_from_directory(base_path)

additional_data_path = "E:/zlx2/pythonProject1/data/guiji_data/高价值目标识别推理/M5"
additional_data = load_data_from_subfolder(additional_data_path, 3, 2)
additional_labels = [max(labels) + 1] * len(additional_data)

data.extend(additional_data)
labels.extend(additional_labels)

torch.save((data, labels), 'E:/zlx2/pythonProject1/data/timeseries_dataset/dataset.pth')
