import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import read_long_lat
import os


def generate_trajectory_map_and_save(lon, lat, filename):
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(224/80, 224/80), dpi=80)  # 224/80 是为了生成224x224大小的图

    # 根据轨迹坐标自动调整地图范围
    center_lon = (min(lon) + max(lon)) / 2
    center_lat = (min(lat) + max(lat)) / 2
    range_lon = max(lon) - min(lon)
    range_lat = max(lat) - min(lat)

    # 使用更大的范围作为范围，确保轨迹完全在图中
    extent_range = max(range_lon, range_lat) / 2 + 0.01

    ax.set_extent([center_lon - extent_range, center_lon + extent_range,
                   center_lat - extent_range, center_lat + extent_range])

    # 添加地图特性
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.LAKES, edgecolor='black')
    ax.add_feature(cfeature.RIVERS)

    # 绘制轨迹
    ax.plot(lon, lat, 'b-', transform=ccrs.Geodetic())

    # 标记轨迹的起点和终点
    ax.plot(lon[0], lat[0], 'go', markersize=5, transform=ccrs.Geodetic())
    ax.plot(lon[-1], lat[-1], 'ro', markersize=5, transform=ccrs.Geodetic())

    # 保存地图为PNG文件
    plt.tight_layout()
    plt.savefig(filename, dpi=80)
    plt.show()



def generate_pic(folder_path, aim_path):
    data = read_long_lat.extract_coordinates(folder_path)

    for idx, (lat, lon) in enumerate(data):
        # 判断这一组经纬度是否超过正常范围
        if (min(lon) < -180 or max(lon) > 180) or (min(lat) < -90 or max(lat) > 90):
            print(f"数据组 {idx + 1} 的经纬度超出范围，跳过该组!")
            continue  # 跳过当前数据组，并继续下一组

        print(lon)
        print(lat)
        print(min(lon), max(lon), min(lat), max(lat))

        # 使用os.path.join确保文件路径正确
        output_filepath = os.path.join(aim_path, f'trajectory_map{idx + 1}.png')
        generate_trajectory_map_and_save(lon, lat, output_filepath)



def generate_pic_for_all_folders(root_folder, output_root):
    """
    :param root_folder: 主目录，其中包含要处理的所有子文件夹。
    :param output_root: 输出的主目录，用于保存生成的图片。
    """
    # 使用os.walk遍历root_folder及其所有子文件夹
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            folder_path = os.path.join(dirpath, dirname)  # 完整的子文件夹路径

            # 根据子文件夹的名字创建输出文件夹
            output_folder = os.path.join(output_root, dirname)

            # 确保输出文件夹存在
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # 对该子文件夹应用generate_pic功能
            generate_pic(folder_path, output_folder)


# 调用函数
root_folder = 'H:/pythonProject1/data/guiji_data/高价值目标识别推理'  # 替换为你的主目录路径
output_root = 'H:/pythonProject1/data/pic'  # 替换为你的输出目录路径
generate_pic_for_all_folders(root_folder, output_root)

