import cv2
import numpy as np
import os
# 可视化
# 输入和输出文件夹路径
input_folder = 'test/images'
output_folder = 'output/images'
labels_folder = 'runs/detect/yolov10/labels'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 坐标转换，原始存储的是YOLOv5格式
def xywh2xyxy(x, y, w, h, img_width, img_height, img):
    # 边界框反归一化
    x_t = int(x * img_width)
    y_t = int(y * img_height)
    w_t = int(w * img_width)
    h_t = int(h * img_height)
    
    # 计算左上角和右下角坐标
    top_left_x = x_t - w_t // 2
    top_left_y = y_t - h_t // 2
    bottom_right_x = x_t + w_t // 2
    bottom_right_y = y_t + h_t // 2
    
    # 绘图，注意rectangle()函数需要坐标为整数
    cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
    return img

# 处理单个图像和对应的标签文件
def process_image(image_path, label_path):
    # 读取图像文件
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image {image_path}")
        return
    
    img_height, img_width = img.shape[:2]
    
    # 检查标签文件是否存在
    if not os.path.isfile(label_path):
        print(f"Warning: No label file found for {image_path}. Skipping.")
        return
    
    # 读取标签文件
    with open(label_path, 'r') as f:
        lines = f.read().strip().splitlines()
        lb = np.array([x.split() for x in lines], dtype=np.float32)  # labels
    
    # 遍历标签并绘制矩形框
    for label in lb:
        if len(label) > 1:  # 确保有足够的数据（至少包含类别和坐标）
            x, y, width, height = map(float, label[1:])  # 将坐标转换为浮点数
            img = xywh2xyxy(x, y, width, height, img_width, img_height, img)
    
    # 输出处理后的图像
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    print(f"Processed and saved {output_path}")

# 遍历输入文件夹中的所有图像文件
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        image_path = os.path.join(input_folder, filename)
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(labels_folder, label_filename)
        process_image(image_path, label_path)