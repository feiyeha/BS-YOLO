from ultralytics import YOLO
import cv2
import torch
import numpy as np
import os
import time
from torchvision import transforms
from PIL import Image
import shutil
from src import GRFBUNet
# 这个文件是输入图像给YOLO和unet后，得到目标检测图和，语义分割图的，然后判定是否违规
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()
# 坐标转换，原始存储的是YOLOv5格式
def xywh2xyxy(x_center, y_center, width, height, img_width, img_height):
    x_min = int((x_center - width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    x_max = int((x_center + width / 2) * img_width)
    y_max = int((y_center + height / 2) * img_height)
    return x_min, y_min, x_max, y_max

# 读取与图像同名的txt文件中的目标检测框坐标
def read_detection_boxes(image_name, detection_dir):
    detection_file = os.path.join(detection_dir, f"{image_name}.txt")
    if not os.path.exists(detection_file):
        raise FileNotFoundError(f"Detection file for {image_name} not found in {detection_dir}")
    with open(detection_file, 'r') as f:
        lines = f.readlines()
    detection_boxes = [tuple(map(float, line.strip().split())) for line in lines]
    return detection_boxes

# 判断是否存在违停行为
def is_parking_violation(detection, blind_way_mask, img_height, img_width):
    category_id, x_center, y_center, width, height = detection
    x_min, y_min, x_max, y_max = xywh2xyxy(x_center, y_center, width, height, img_width, img_height)

    # 确保检测框在图像范围内
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(img_width, x_max), min(img_height, y_max)

    # 提取检测框内的盲道分割结果，一个原图的，一个全局图的
    roi_mask = blind_way_mask[y_min:y_max, x_min:x_max]

#     现在盲道分割结果是从原图上进行分割的，我接下来要实现框内盲道像素点数量和全局图框内盲道像素点数量之比,该方法不需要分割其他的像素，如汽车像素,步骤是:
# 首先应该获得全局图，通过输入或者高斯建模
# 首先框要映射到原图和全局图的分割结果上,然后从原图分割结果上得到框内盲道像素点数量,
#还有从全局图分割结果中,得到全局图框内盲道像素点数量。相除乘100%

    #统计原图和全局图框内盲道像素点数量
    pixel_count = np.sum(roi_mask == 255)
# 这的不用背景图方法：用盲道与人行道的像素数之比，人行道像素是包括盲道在内的，就是看框内，所有的道路像素点，盲道占所有道路像素点的百分比
    # # 如果检测框内包含盲道像素点，则判定为违停（这里假设类别号为0表示车辆）
    if  pixel_count>100:
        return True, (x_min, y_min, x_max, y_max)  # 返回True和违停框坐标
    return False, None

# def is_parking_violation(detection, blind_way_mask, blind_way_mask2, img_height, img_width):
#     category_id, x_center, y_center, width, height = detection
#     x_min, y_min, x_max, y_max = xywh2xyxy(x_center, y_center, width, height, img_width, img_height)
#
#     # 确保检测框在图像范围内
#     x_min, y_min = max(0, x_min), max(0, y_min)
#     x_max, y_max = min(img_width, x_max), min(img_height, y_max)
#
#     # 提取检测框内的盲道分割结果，一个原图的，一个全局图的
#     roi_mask = blind_way_mask[y_min:y_max, x_min:x_max]
#     roi_mask2 = blind_way_mask2[y_min:y_max, x_min:x_max]
#
#     #     现在盲道分割结果是从原图上进行分割的，我接下来要实现框内盲道像素点数量和全局图框内盲道像素点数量之比,该方法不需要分割其他的像素，如汽车像素,步骤是:
#     # 首先应该获得全局图，通过输入或者高斯建模
#     # 首先框要映射到原图和全局图的分割结果上,然后从原图分割结果上得到框内盲道像素点数量,
#     # 还有从全局图分割结果中,得到全局图框内盲道像素点数量。相除乘100%
#
#     # 统计原图和全局图框内盲道像素点数量
#     pixel_count = np.sum(roi_mask == 255)
#     pixel_count2 = np.sum(roi_mask2 == 255)
#     # 可以理解成计算盲道被遮挡的比例的，也可以理解成增加了多少像素点的占比70,1000
#     # 公式一：（全局图框内盲道像素点数量-原图框内盲道像素点数量）/全局图框内盲道像素点数量=(1000-70)/1000=0.93
#     # 公式二：1-（原图框内盲道像素点数量/全局图框内盲道像素点数量）=1-（70/1000）=1-0.07=0.93这两个本质是一样的
#     if pixel_count2 > 0:
#         occupy = 1 - (pixel_count / pixel_count2)
#     else:
#         occupy = 0
#     # # 如果检测框内包含盲道像素点，则判定为违停（这里假设类别号为0表示车辆）
#     print(pixel_count)
#     print(pixel_count2)
#     print(occupy)
#     # 也就是盲道像素点数量变化差距越大，occupy就会越大，越大准确率越高，也就是增了百分之10的像素点就可以判定，
#     if occupy >= 0.1:
#         return True, (x_min, y_min, x_max, y_max)  # 返回True和违停框坐标
#     return False, (x_min, y_min, x_max, y_max)
# 处理单个图像和对应的标签及分割文件
def process_image(image_path, detection_path, segmentation_path,output_dir,output_dir2):
    img = cv2.imread(image_path)
    segimg = cv2.imread(segmentation_path)
    # backsegimg=cv2.imread(segmentation_path2)
    if img is None:
        print(f"Error: Unable to read image {image_path}")
        return
    
    img_height, img_width = img.shape[:2]
    total_boxes = 0
    violation_boxes_count = 0
    normal_boxes_count = 0

    try:
#         os.path.splitext(os.path.basename(image_path))[0]获取文件名，不包含扩展名，/home/user/pictures/photo.jpg，得到photo
        detection_boxes = read_detection_boxes(os.path.splitext(os.path.basename(image_path))[0],os.path.dirname(detection_path))
    except FileNotFoundError:
        print(f"Warning: No detection file found for {image_path}. Skipping.")
        return
    if not os.path.exists(segmentation_path):
        print(f"Warning: No segmentation file found for {image_path}. Skipping.")
        return
#     表示以灰度模式读取分割图，即图像将被转换为灰度图，只包含亮度信息，而不包含颜色信息。这对于图像分割、边缘检测等任务来说非常有用，因为它们通常不需要颜色信息。
    blind_way_mask = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE)
    # blind_way_mask2= cv2.imread(segmentation_path2, cv2.IMREAD_GRAYSCALE)
    violation_found = False
    violation_boxes = []
    
    for detection in detection_boxes:
        total_boxes += 1  # 总检测框数加1
#         输入框，分割图，图像宽高，判断每个框内是否违规
        violation, violation_box = is_parking_violation(detection, blind_way_mask,img_height, img_width)
        if violation:
            violation_found = True
            violation_boxes.append(violation_box)
            # 绘制违停框（红色）
            cv2.rectangle(segimg, violation_box[:2], violation_box[2:], (0, 0, 255), 2)
            violation_boxes_count += 1
            cv2.rectangle(img, violation_box[:2], violation_box[2:], (0, 0, 255), 2)
            # cv2.rectangle(backsegimg, violation_box[:2], violation_box[2:], (0, 0, 255), 2)
        else:
            # 绘制正常检测框（绿色）
            normal_boxes_count += 1
            x_min, y_min, x_max, y_max = xywh2xyxy(*detection[1:5], img_width, img_height)
            cv2.rectangle(segimg, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # cv2.rectangle(backsegimg, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
      # 添加文本说明（可选）
    if violation_found:
        cv2.putText(img, 'Violation Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        violation_status = "Violation"

    else:
        violation_status = "No Violation"
    
    # 保存处理后的图像，并输出是否违停的信息
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    output_path2 = os.path.join(output_dir2, os.path.basename(image_path))
    cv2.imwrite(output_path2, img)
    cv2.imwrite(output_path, segimg)
    # cv2.imwrite(output_path, backsegimg)
    # 输出统计信息
    print(f"Processed and saved {output_path} - {violation_status}")
    print(f"当前Detection Summary: 总框: {total_boxes}, 违停 (Red): {violation_boxes_count}, 不违停 (Green): {normal_boxes_count}")
    return total_boxes, violation_boxes_count, normal_boxes_count

def main():
    # 1.获取输入图像
    image_path = 'test/images'
    output_dir='runs/detect/predict'
    # 2.使用YOLO对输入图像进行目标检测，得到汽车和非机动车的识别框。

    # model = YOLOv10('weights/best.pt')
    # results=model(image_path,show=True,save=True)
    #
    model = YOLO(model=r'weights/AMFA+ALEA.pt')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    model.predict(source=image_path,
                  save=True,
                  show=False,
                  )

    # # 3.语义分割模型对输入图像进行分割，得到盲道的分割结果，还要同时输入全局图进行分割。
    # classes = 1  # exclude background
    # weights_path = "weights/model_best.pth"
    # img_path =image_path
    # txt_path = "test/index/predict.txt"
    # save_result = "predict/Part01"
    #
    # assert os.path.exists(weights_path), f"weights {weights_path} not found."
    # assert os.path.exists(img_path), f"image {img_path} not found."
    #
    # mean = (0.709, 0.381, 0.224)
    # std = (0.127, 0.079, 0.043)
    #
    # # get devices
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("using {} device.".format(device))
    #
    # # create model
    # model = GRFBUNet(in_channels=3, num_classes=classes+1, base_c=32)
    #
    # # load weights
    # model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    # model.to(device)
    #
    #
    # total_time = 0
    # count = 0
    # with open(os.path.join(txt_path), 'r') as f:
    #     file_name = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
    # for file in file_name:
    #     original_img = Image.open(os.path.join(img_path, file + ".jpg")).convert('RGB')
    #     count = count +1
    #     h = np.array(original_img).shape[0]
    #     w = np.array(original_img).shape[1]
    #
    #
    #
    #     data_transform = transforms.Compose([transforms.Resize(565),
    #                                      transforms.ToTensor(),
    #                                      transforms.Normalize(mean=mean, std=std)])
    #     img = data_transform(original_img)
    #     # expand batch dimension
    #
    #     img = torch.unsqueeze(img, dim=0)
    #
    #     model.eval()  # Entering Validation Mode
    #     with torch.no_grad():
    #         # init model
    #         img_height, img_width = img.shape[-2:]
    #         init_img = torch.zeros((1, 3, img_height, img_width), device=device)
    #         model(init_img)
    #
    #         t_start = time_synchronized()
    #         output = model(img.to(device))
    #         t_end = time_synchronized()
    #         total_time = total_time + (t_end - t_start)
    #         print("inference+NMS time: {}".format(t_end - t_start))
    #
    #         prediction = output['out'].argmax(1).squeeze(0)
    #         prediction = prediction.to("cpu").numpy().astype(np.uint8)
    #         prediction = cv2.resize(prediction, (w, h), interpolation = cv2.INTER_LINEAR)
    #         # Change the pixel value corresponding to the foreground to 255 (white)
    #         prediction[prediction == 1] = 255
    #         # Set the pixels in the area of no interest to 0 (black)
    #         prediction[prediction == 0] = 0
    #         mask = Image.fromarray(prediction)
    #         mask = mask.convert("L")
    #         name = file[-4:]
    #
    #         if not os.path.exists(save_result):
    #             os.makedirs(save_result)
    #
    #         mask.save(os.path.join(save_result, f'{name}.png'))
    # fps = 1 / (total_time / count)
    # print("FPS: {}".format(fps))
    # 示例使用，接下来上面路径输入时为了语义分割得到结果，下面的路径是输入，然后处理是否违规
    # 假设 image_dir 是包含输入图像的目录路径
    # detection_dir 是包含txt文件的目录路径（与图像同名）
    # segmentation_dir 是包含语义分割结果的目录路径（与图像同名，但可能是不同的文件格式）
    # 输入和输出文件夹路径
    # image_dir = 'test/images'
    # output_dir = 'output/resultmask'
    # output_dir2 = 'output/result'
    # detection_dir = 'runs/detect/predict/labels'
    # segmentation_dir = 'predict/Part01'
    # # quanjusegmentation_dir = 'predict/Part02'
    # total_boxes = 0
    # violation_boxes_count = 0
    # normal_boxes_count = 0
    # # 确保输出文件夹存在
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # if not os.path.exists(output_dir2):
    #     os.makedirs(output_dir2)
    # # 遍历输入文件夹中的所有图像文件
    # for filename in os.listdir(image_dir):
    #     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
    #         image_path = os.path.join(image_dir, filename)
    #         # image_path:test/images加上图像名组成图像文件的路径
    #         detection_filename = os.path.splitext(filename)[0] + '.txt'
    #        #预测的标签文件加上图像名组成标签文件的路径
    #         detection_path = os.path.join(detection_dir, detection_filename)
    #         segmentation_filename = os.path.splitext(filename)[0] + '.png'
    #         #图像分割盲道的文件加上图像名组成分割文件的路径
    #         segmentation_path = os.path.join(segmentation_dir, segmentation_filename)
    #         #图像分割盲道的文件加上图像名组成分割文件的路径全局图的
    #         # segmentation_path2 = os.path.join(quanjusegmentation_dir, segmentation_filename)
    #         total, violation, normal=process_image(image_path, detection_path, segmentation_path,output_dir,output_dir2)
    #         total_boxes=total_boxes+total
    #         violation_boxes_count=violation_boxes_count+violation
    #         normal_boxes_count=normal+normal_boxes_count
    # print(f"所有图片的Detection Summary: 总框: {total_boxes}, 违停 (Red): {violation_boxes_count}, 不违停 (Green): {normal_boxes_count}")

if __name__ == '__main__':
    main()



