
import os
import csv


def create_csv_with_fixed_label(data_folder, csv_file, fixed_label=1):
    # 获取所有图像文件名
    image_files = os.listdir(data_folder)
    image_files = [filename for filename in image_files if filename.endswith('.png')]

    # 创建CSV文件并写入标题
    with open(csv_file, mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['image_path', 'label'])  # 写入标题

        # 遍历图像文件并将固定标签写入CSV
        for filename in image_files:
            image_path = os.path.join(data_folder, filename)
            csvwriter.writerow([image_path, fixed_label])

    print("CSV file creation completed.")



data_dir = 'E:/zlx2/pythonProject1/data/pic'
csv_file = '/data/csv_dataset/dataset.csv'

class_folders = ['M5', 'hangji1', 'hangji2', 'hangji3', 'hangji4']


with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'label'])

with open(csv_file, 'a', newline='') as f:
    writer = csv.writer(f)
    label = 0
    for class_folder in class_folders:
        class_dir = os.path.join(data_dir, class_folder)
        for filename in os.listdir(class_dir):
            if filename.endswith('.png'):
                image_path = os.path.join(class_dir, filename)
                writer.writerow([image_path, label])
        label += 1

# # 调用函数并传入数据文件夹和CSV文件路径
# data_folder = 'H:/pythonProject1/data/pic/M5'
# csv_file ='H:/pythonProject1/data/csv_dataset/M5.csv'
# create_csv_with_fixed_label(data_folder, csv_file)




