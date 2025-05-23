import os


def get_image_names_to_txt(input_dir, output_txt):
    """
    获取指定目录下所有图像文件名（不带扩展名）并写入txt文件

    参数:
        input_dir (str): 包含图像的目录路径
        output_txt (str): 要输出的txt文件路径
    """
    # 支持的图像文件扩展名（不区分大小写）
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    # 收集所有符合条件的文件名
    image_names = []

    for filename in os.listdir(input_dir):
        # 获取文件扩展名并转为小写
        file_ext = os.path.splitext(filename)[1].lower()

        # 如果是图像文件
        if file_ext in image_extensions:
            # 获取不带扩展名的文件名
            base_name = os.path.splitext(filename)[0]
            image_names.append(base_name)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)

    # 写入txt文件
    with open(output_txt, 'w', encoding='utf-8') as f:
        for name in image_names:
            f.write(f"{name}\n")

    print(f"成功写入 {len(image_names)} 个图像名到 {output_txt}")


# 使用示例
if __name__ == "__main__":
    # 替换为你的实际路径
    image_directory = 'test/images'  # 图像所在目录
    output_file = 'test/index/predict.txt'  # 输出文件路径

    get_image_names_to_txt(image_directory, output_file)