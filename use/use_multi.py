import torch
from torchvision import transforms
from PIL import Image
from model import pvt_reduce  # 确保这个导入语句与你的模型路径匹配
from model import multi_model
# 加载模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = multi_model.pvt_tiny(num_classes=5, pretrained=False)  # 使用你的模型
model.load_state_dict(torch.load('E:/zlx2/pythonProject1/save_model/pvt_tiny_5_classes.pth'))  # 加载训练好的权重
model = model.to(device)
model.eval()  # 将模型设置为评估模式

# 定义图像预处理步骤
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    # 打开图像并应用变换
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # 增加一个批处理维度

    # 进行预测
    with torch.no_grad():
        outputs,feature_maps = model(image)
        _, predicted = torch.max(outputs.data, 1)


    return predicted.item(),feature_maps  # 返回预测的类别

# 使用模型进行预测
image_path= 'E:/zlx2/pythonProject1/data/pic/hangji1/trajectory_map1.png'
predicted_class,feature_maps = predict_image(image_path)
print(f'The predicted class is: {predicted_class}')


