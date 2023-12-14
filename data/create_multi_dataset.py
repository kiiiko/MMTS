import pandas as pd
import torch
from torch.utils.data import Dataset

class CustomMultiModalDataset(Dataset):
    def __init__(self, image_csv_file, time_series_pth_file):
        """
        Args:
            image_csv_file (str): 图像数据和标签的CSV文件路径。
            time_series_pth_file (str): 时间序列数据和标签的.pth文件路径。
        """
        self.image_data = pd.read_csv(image_csv_file)
        self.time_series_data = torch.load(time_series_pth_file)  # 假设这是一个PyTorch张量

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        # 加载图像数据（假设第一列是标签，其他列是图像数据）
        image = self.image_data.iloc[idx, 1:].values.astype('float32')
        image = torch.Tensor(image)

        # 加载时间序列数据
        time_series = self.time_series_data[idx]

        # 加载标签（CSV和.pth文件中的第2列是标签）
        label = self.image_data.iloc[idx, 1]
        label = torch.Tensor([label]).long()

        return image, time_series, label

# 用法示例
image_csv_file = '/data/csv_dataset/dataset.csv'
time_series_pth_file = '/data/timeseries_dataset/dataset.pth'

dataset = CustomMultiModalDataset(image_csv_file, time_series_pth_file)


# 保存到本地
torch.save(dataset, '/data/csv_dataset/multi_dataset.pth')
