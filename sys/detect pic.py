from ultralytics import YOLOv10
import cv2
import torch
import numpy as np
import os
import time
from torchvision import transforms
from PIL import Image
from src import GRFBUNet
# 这个文件是输入图像给YOLO和unet后，得到目标检测图和，语义分割图的，然后判定是否违规
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def main():
    # 1.获取输入图像
    image_path = 'test/images'

    # 2.使用YOLOv11对输入图像进行目标检测，得到汽车和非机动车的识别框。

#     model = YOLOv10('weights/best.pt')
#     results=model(image_path,show=True,save=True)
# yolov11
    model = YOLO(model=r'weights/v11wubest.pt')
    model.predict(source=image_path,
                  save=True,
                  show=True,
                  )

    # 3.语义分割模型对输入图像进行分割，得到盲道的分割结果
    classes = 1  # exclude background
    weights_path = "weights/model_best.pth"
    img_path =image_path
    txt_path = "test/index/predict.txt"
    save_result = "predict/Part01"

    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = GRFBUNet(in_channels=3, num_classes=classes+1, base_c=32)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)


#     total_time = 0
#     count = 0
#     with open(os.path.join(txt_path), 'r') as f:
#         file_name = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
#     for file in file_name:
#         original_img = Image.open(os.path.join(img_path, file + ".jpg")).convert('RGB')
#         count = count +1
#         h = np.array(original_img).shape[0]
#         w = np.array(original_img).shape[1]



#         data_transform = transforms.Compose([transforms.Resize(565),
#                                          transforms.ToTensor(),
#                                          transforms.Normalize(mean=mean, std=std)])
#         img = data_transform(original_img)
#         # expand batch dimension

#         img = torch.unsqueeze(img, dim=0)

#         model.eval()  # Entering Validation Mode
#         with torch.no_grad():
#             # init model
#             img_height, img_width = img.shape[-2:]
#             init_img = torch.zeros((1, 3, img_height, img_width), device=device)
#             model(init_img)

#             t_start = time_synchronized()
#             output = model(img.to(device))
#             t_end = time_synchronized()
#             total_time = total_time + (t_end - t_start)
#             print("inference+NMS time: {}".format(t_end - t_start))

#             prediction = output['out'].argmax(1).squeeze(0)
#             prediction = prediction.to("cpu").numpy().astype(np.uint8)
#             prediction = cv2.resize(prediction, (w, h), interpolation = cv2.INTER_LINEAR)
#             # Change the pixel value corresponding to the foreground to 255 (white)
#             prediction[prediction == 1] = 255
#             # Set the pixels in the area of no interest to 0 (black)
#             prediction[prediction == 0] = 0
#             mask = Image.fromarray(prediction)
#             mask = mask.convert("L")
#             name = file[-4:]

#             if not os.path.exists(save_result):
#                 os.makedirs(save_result)

#             mask.save(os.path.join(save_result, f'{name}.png'))
#     fps = 1 / (total_time / count)
#     print("FPS: {}".format(fps))
    # 示例使用
    # 假设 image_dir 是包含输入图像的目录路径
    # detection_dir 是包含txt文件的目录路径（与图像同名）
    # segmentation_dir 是包含语义分割结果的目录路径（与图像同名，但可能是不同的文件格式）
import os
import cv2
import numpy as np

image_dir = "test/images"
detection_dir = "runs/detect/predict/labels"
segmentation_dir = "predict/Part01"
output = "output"

def xywh2xyxy(x_center, y_center, width, height, img_width, img_height):
    x_t = int(x_center * img_width)
    y_t = int(y_center * img_height)
    w_t = int(width * img_width)
    h_t = int(height * img_height)
    
    top_left_x = x_t - w_t // 2
    top_left_y = y_t - h_t // 2
    bottom_right_x = x_t + w_t // 2
    bottom_right_y = y_t + h_t // 2
    
    return top_left_x, top_left_y, bottom_right_x, bottom_right_y

def read_detection_boxes(image_name, detection_dir):
    detection_file = os.path.join(detection_dir, f"{image_name}.txt")
    if not os.path.exists(detection_file):
        raise FileNotFoundError(f"Detection file for {image_name} not found in {detection_dir}")

    with open(detection_file, 'r') as f:
        lines = f.readlines()
    
    detection_boxes = []
    for line in lines:
        parts = line.strip().split()
        category_id = int(parts[0])
        x_center_norm = float(parts[1])
        y_center_norm = float(parts[2])
        width_norm = float(parts[3])
        height_norm = float(parts[4])
        detection_boxes.append((category_id, x_center_norm, y_center_norm, width_norm, height_norm))
    
    return detection_boxes

def is_parking_violation(image, detection_boxes, blind_way_mask):
    img_width, img_height = image.shape[:2]
    
    for detection in detection_boxes:
        category_id, x_center_norm, y_center_norm, width_norm, height_norm = detection
        
        x_min, y_min, x_max, y_max = xywh2xyxy(x_center_norm, y_center_norm, width_norm, height_norm, img_width, img_height)
        
        # 确保检测框在图像范围内
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(img_width, x_max), min(img_height, y_max)
        
        # 提取检测框内的盲道分割结果
        roi_mask = blind_way_mask[y_min:y_max, x_min:x_max]
        
        # 如果检测框内包含盲道像素点，则判定为违停（这里假设类别号为0表示汽车）
        if category_id == 0 and np.any(roi_mask == 1):
            return True
    
    return False

# 遍历图像文件
for image_name in os.listdir(image_dir):
    if image_name.endswith(".jpg") or image_name.endswith(".png"):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        
        try:
            detection_boxes = read_detection_boxes(image_name.split(".")[0], detection_dir)
        except FileNotFoundError:
            print(f"Image: {image_name}, No detection file found. No violation detected.")
            continue
        
        segmentation_path = os.path.join(segmentation_dir, f"{image_name.split('.')[0]}.png")
        if not os.path.exists(segmentation_path):
            print(f"Image: {image_name}, No segmentation file found. No violation detected.")
            continue
        
        blind_way_mask = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE)
        
        violation = is_parking_violation(image, detection_boxes, blind_way_mask)
        print(f"Image: {image_name}, Violation: {violation}")
#     image_dir = "test/images"
#     detection_dir = "runs/detect/predict/labels"
#     segmentation_dir = "predict/Part01"
#     output="output"
#     # 遍历图像文件
#     for image_name in os.listdir(image_dir):
#         if image_name.endswith(".jpg") or image_name.endswith(".png"):  # 根据你的图像格式进行调整
#             image_path = os.path.join(image_dir, image_name)
#             image = cv2.imread(image_path)
        
#         # 尝试读取目标检测框坐标
#             try:
#                 detection_boxes = read_detection_boxes(image_name.split(".")[0], detection_dir)
#             except FileNotFoundError:
#             # 如果检测框文件不存在，则输出未检测到违规并继续下一张图片
#                 print(f"Image: {image_name}, No detection file found. No violation detected.")
#                 continue  # 跳过当前循环，继续下一张图片
        
#             # 读取盲道分割结果（这里假设分割结果是与图像同名的二值图像文件）
#             segmentation_path = os.path.join(segmentation_dir, f"{image_name.split('.')[0]}.png")
#             if not os.path.exists(segmentation_path):
#             # 如果分割结果文件也不存在，则同样输出未检测到违规并继续
#                 print(f"Image: {image_name}, No segmentation file found. No violation detected.")
#                 continue
        
#             blind_way_mask = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE)
        
#             # 判断是否存在违停行为
#             violation = is_parking_violation(image, detection_boxes,output)
        
#             # 输出结果
#             print(f"Image: {image_name}, Violation: {violation}")
 
            
# # 4判断违停，将识别框对应的区域，坐标定位在语义分割结果中找到，这个坐标内是否包含盲道像素点。如果包含，则判定为违停
# # 这代码转换的有问题
# def read_detection_boxes(image_name, detection_dir):
#     """
#     读取与图像同名的txt文件中的目标检测框坐标。
    
#     参数:
#     image_name (str): 图像的文件名（不包含路径和后缀）。
#     detection_dir (str): 包含txt文件的目录路径。
    
#     返回:
#     list of tuples: 目标检测框坐标列表，每个坐标为一个元组 (category_id, x_min, y_min, x_max, y_max)。
#     """
#     detection_file = os.path.join(detection_dir, f"{image_name}.txt")
#     if not os.path.exists(detection_file):
#         raise FileNotFoundError(f"Detection file for {image_name} not found in {detection_dir}")

#     with open(detection_file, 'r') as f:
#         lines = f.readlines()
    
#     detection_boxes = []
#     for line in lines:
#         parts = line.strip().split()
#         category_id = int(parts[0])
#         x_min_norm = float(parts[1])
#         y_min_norm = float(parts[2])
#         x_max_norm = float(parts[3])
#         y_max_norm = float(parts[4])
        
#         # 这里假设你有一个函数可以获取图像的尺寸
#         # image_shape = get_image_shape(image_name)  # 应该是 (height, width)
#         # 但由于你提到每张图像的大小尺寸都不太一样，并且不可能手动输入，
#         # 我们需要在处理每张图像时动态地加载它来获取尺寸。
#         # 然而，在这个函数中我们还没有图像数据，所以我们暂时不处理具体尺寸，
#         # 而是将归一化坐标作为输出的一部分，供后续处理使用。
        
#         # 你可以选择在后续处理中传递图像尺寸，或者在这里加载图像以获取尺寸。
#         # 但为了保持函数的独立性，这里我们仅输出归一化坐标。
        
#         detection_boxes.append((category_id, x_min_norm, y_min_norm, x_max_norm, y_max_norm))
    
#     return detection_boxes
# def is_parking_violation(image, detection_boxes, blind_way_mask):
#     """
#     判断是否存在违停行为。
    
#     参数:
#     image (numpy.ndarray): 输入图像，用于获取图像的尺寸。
#     detection_boxes (list of tuples): 目标检测框坐标列表，每个坐标为一个元组 (category_id, x_min_norm, y_min_norm, x_max_norm, y_max_norm)。
#     blind_way_mask (numpy.ndarray): 盲道分割结果，与输入图像尺寸相同。
    
#     返回:
#     bool: 是否存在违停行为。
#     """
#     img_width, img_height = image.shape[:2]  # 获取图像的尺寸 (height, width)
    
#     for detection in detection_boxes:
#         category_id, x_min_norm, y_min_norm, x_max_norm, y_max_norm = detection
#         img = xywh2xyxy(x, y, width, height, img_width, img_height, img)
       
#         # 确保检测框在图像范围内
#         x_min, y_min = max(0, x_min), max(0, y_min)
#         x_max, y_max = min(image_shape[1], x_max), min(image_shape[0], y_max)
        
#         # 提取检测框内的盲道分割结果
#         roi_mask = blind_way_mask[y_min:y_max, x_min:x_max]
        
#         # 如果检测框内包含盲道像素点，则判定为违停（这里假设类别号为0表示汽车或非机动车）
#         # 你可以根据实际需求调整类别号的判断逻辑
#         if category_id == 0 and np.any(roi_mask == 1):
#             return True
    
#     return False

if __name__ == '__main__':
    main()



