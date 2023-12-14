import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split
from model import pvt # 导入你
from model import pvt_reduce
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from timm.models.layers import trunc_normal_
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)  # 假设CSV文件包含 'image_path' 和 'label' 列
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]  # 图像文件路径
        image = Image.open(img_path).convert('RGB')  # 以RGB格式打开图像
        label = self.data.iloc[idx, 1]  # 图像标签

        if self.transform:
            image = self.transform(image)

        return image, label

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义一些训练相关的参数
batch_size =32
learning_rate = 0.001
num_epochs = 150

# 创建数据集实例
full_dataset = CustomDataset('E:/zlx2/pythonProject1//data/csv_dataset/dataset.csv', transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), # 将图像转为Tensor格式
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用ImageNet数据集的均值和标准差

]))

# 划分数据集为训练集和验证集
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型
model = pvt_reduce.pvt_tiny(num_classes=5, pretrained=False)

#使用初始化参数
model.apply(init_weights)

#将模型传输到设备
model = model.to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 定义学习率调度器
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
writer = SummaryWriter('/runs/training_loss_visualization')

# 训练循环
global_step = 0  # 这将帮助我们记录整体的步骤
for epoch in range(num_epochs):
    model.train()

    # 初始化变量来记录整个epoch的loss
    total_loss = 0
    num_batches = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 累加loss值和batch数
        total_loss += loss.item()
        num_batches += 1

        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{global_step}], Train Loss: {loss.item():.4f}')
        global_step += 1

    # 在每个epoch结束后，计算并记录平均loss
    avg_train_loss = total_loss / num_batches
    writer.add_scalar('Training Loss/epoch', avg_train_loss, epoch)

    # 在每个epoch结束后进行验证
    model.eval()
    scheduler.step()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    writer.add_scalar('Validation Accuracy', val_accuracy, epoch)
    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Avg Train Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

writer.close()
print('Training finished')

model_save_path = '/save_model/pvt_tiny_5_classes_925.pth'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')
