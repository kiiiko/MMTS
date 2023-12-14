from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import pandas as pd
import os
import torch
from model import transformer_model
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])


def pad_and_mask(batch):
    # 获取外部和内部维度的最大长度
    max_length_outer = max([seq.shape[0] for seq, _ in batch])
    max_length_inner = max([seq.shape[1] for seq, _ in batch])

    padded_inputs = []
    masks = []

    for (seq, label) in batch:
        # 对外部维度进行填充
        padding_outer = max_length_outer - seq.shape[0]
        # 对内部维度进行填充
        padding_inner = max_length_inner - seq.shape[1]

        # 为每个序列添加填充
        padded_seq = F.pad(seq, (0, padding_inner, 0, padding_outer))
        padded_inputs.append(padded_seq)

        # 创建外部和内部掩码
        mask_outer = [[1] * seq.shape[1] + [0] * padding_inner] * seq.shape[0] + [
            [0] * max_length_inner] * padding_outer
        masks.append(torch.tensor(mask_outer))

    labels = [label for _, label in batch]

    return torch.stack(padded_inputs), torch.stack(masks), torch.tensor(labels)

d_model = 512  # 设置合适的值
n_layers = 1   # 设置合适的值

data, labels = torch.load('E:/zlx2/pythonProject1/data/timeseries_dataset/dataset.pth')
total_dataset = CustomDataset(data, labels)
total_size = len(total_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
from torch.utils.data import random_split

train_dataset, val_dataset = random_split(total_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_and_mask)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=pad_and_mask)


# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
input_dim = 2
num_classes = len(set(labels))
model = transformer_model.Transformer(d_model=d_model, n_layers=n_layers, n_heads=8,device=device).to(device)
learning_rate = 0.001
epochs = 10000
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCEWithLogitsLoss()

def train():
    model.train()
    total_loss = 0
    for batch_idx, (data, mask,label) in enumerate(train_loader):
        data = data.float().to(device)  # 为data设置dtype并分配设备
        label = label.long().to(device)  # 为target设置dtype并分配设备

        mask = mask.to(device)



        optimizer.zero_grad()
        output, _ = model(data.to(device))  # 32 yiwei
        # print(output.shape)
        # print(label.shape)

        loss = criterion(output,label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 每个batch都打印进度
        print('Train [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

    avg_loss = total_loss / len(train_loader)
    print('====> Average loss: {:.4f}'.format(avg_loss))


def validate():
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, mask, label in val_loader:
            data = data.float().to(device)  # 为data设置dtype并分配设备
            label = label.long().to(device)  # 为target设置dtype并分配设备

            mask = mask.to(device)



            output, _ = model(data.to(device))  # 将data移动到模型所在的设备

            val_loss += criterion(output, label).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    val_loss /= len(val_loader)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset), 100. * correct / len(val_loader.dataset)))

    return val_loss



# 主训练循环
best_val_loss = float('inf')
for epoch in range(1, epochs + 1):
    train()
    val_loss = validate()

    # 简单的模型保存（保存最佳模型）
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_transformer_model.pth')