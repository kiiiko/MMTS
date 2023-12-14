from PIL import Image
import os


def resize_image_to_224x224(input_image_path, output_image_path):
    """
    Resize a single image to 224x224 dimensions.

    Parameters:
    - input_image_path: Path to the input image.
    - output_image_path: Path to save the resized image.

    Returns:
    None. The resized image is saved to output_image_path.
    """

    with Image.open(input_image_path) as img:
        img_resized = img.resize((224, 224), Image.BILINEAR)
        img_resized.save(output_image_path)


def batch_resize_images(input_directory, output_directory):
    """
    Resize all images in the input directory and save to the output directory.

    Parameters:
    - input_directory: Directory containing images to resize.
    - output_directory: Directory to save resized images.

    Returns:
    None. All resized images are saved to the output directory.
    """

    # 如果输出目录不存在，则创建它
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 遍历输入目录中的每个文件
    for filename in os.listdir(input_directory):
        # 检查文件是否为图像（这里简单地检查文件扩展名）
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            input_image_path = os.path.join(input_directory, filename)
            output_image_path = os.path.join(output_directory, filename)
            resize_image_to_224x224(input_image_path, output_image_path)
            print(f"Resized: {filename}")

# 使用示例
# batch_resize_images("path_to_input_directory", "path_to_output_directory")
