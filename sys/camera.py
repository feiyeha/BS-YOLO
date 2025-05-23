# ai写的
# import cv2
# import torch
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# from models.unet import GRFBUNet  # 假设你有一个UNet模型
# # 实时摄像头部分#
# # 使用YOLO对开始摄像头实时进行目标检测，得到汽车和非机动车的识别框。打开摄像头使用YOLO监控实时检测，检测到汽车，就把那一帧截下来，进行实时语义分割，把框的坐标对应到这个语义分割图中和背景的语义分割图中，得到对应位置盲道的像素点数量，两个图的像素点数量相除
# def real_time_detection_and_segmentation(background_mask_path, output_dir):
#     # 加载YOLO模型
#     yolo_model = YOLO(model='weights/v11wubest.pt')

#     # 加载语义分割模型
#     classes = 1  # exclude background
#     weights_path = "weights/model_best.pth"
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     segmentation_model = GRFBUNet(in_channels=3, num_classes=classes+1, base_c=32)
#     segmentation_model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
#     segmentation_model.to(device)
#     segmentation_model.eval()

#     # 打开摄像头
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return

#     # 加载背景图的语义分割结果
#     background_mask = Image.open(background_mask_path).convert('L')
#     background_mask = np.array(background_mask)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to capture image.")
#             break

#         # 使用YOLO进行目标检测
#         results = yolo_model.predict(source=frame, save=False, show=False)

#         # 遍历检测结果
#         for result in results:
#             if result['class'] == 'car':  # 假设检测到的类别为'car'
#                 x1, y1, x2, y2 = result['bbox']  # 获取检测框的坐标

#                 # 截取检测框内的图像
#                 car_frame = frame[y1:y2, x1:x2]

#                 # 对截取的图像进行语义分割
#                 car_frame_pil = Image.fromarray(cv2.cvtColor(car_frame, cv2.COLOR_BGR2RGB))
#                 data_transform = transforms.Compose([
#                     transforms.Resize(565),
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=(0.709, 0.381, 0.224), std=(0.127, 0.079, 0.043))
#                 ])
#                 car_frame_tensor = data_transform(car_frame_pil).unsqueeze(0).to(device)

#                 with torch.no_grad():
#                     output = segmentation_model(car_frame_tensor)
#                     prediction = output['out'].argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
#                     prediction = cv2.resize(prediction, (car_frame.shape[1], car_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
#                     prediction[prediction == 1] = 255
#                     prediction[prediction == 0] = 0

#                 # 获取当前帧的语义分割结果
#                 current_mask = prediction

#                 # 计算盲道的像素点比例
#                 background_region = background_mask[y1:y2, x1:x2]
#                 blind_pixel_count_current = np.sum(current_mask == 255)
#                 blind_pixel_count_background = np.sum(background_region == 255)

#                 if blind_pixel_count_background > 0:
#                     blind_ratio = blind_pixel_count_current / blind_pixel_count_background
#                     print(f"Blind road ratio: {blind_ratio:.2f}")

#                 # 保存结果
#                 if not os.path.exists(output_dir):
#                     os.makedirs(output_dir)
#                 cv2.imwrite(os.path.join(output_dir, f'car_{time.time()}.jpg'), frame)
#                 cv2.imwrite(os.path.join(output_dir, f'mask_{time.time()}.png'), current_mask)

#         # 显示实时画面
#         cv2.imshow('Real-time Detection', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # 释放摄像头并关闭窗口
#     cap.release()
#     cv2.destroyAllWindows()


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


# test检测的图像和视频
# background生成的背景图
# predict/part01直接语义分割test的图像
# output/result框花在predict/part01上，语义分割的图上
# runs/detect/predict yolo检测的结果，txt标签
# results:存放检测到违停的帧
# 此文件对应调用视频监控，首先在对监控进行识别时，
# 这些都是提前做的，首先得到背景图
# 1.输入背景图
# 2。输入视频经过混合高斯算法得到背景图
# 得到背景图之后，unet进行语义分割，得到语义分割图。
# 然后就可以开始正规实时的操作了，打开摄像头使用YOLO监控实时检测，检测到汽车，就把那一帧截下来，进行实时语义分割，把框的坐标对应到这个语义分割图中和背景的语义分割图中，得到对应位置盲道的像素点数量，两个图的像素点数量相除
# 判断是否存在违停行为
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def xywh2xyxy(x_center, y_center, width, height, img_width, img_height):
    x_min = int((x_center - width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    x_max = int((x_center + width / 2) * img_width)
    y_max = int((y_center + height / 2) * img_height)
    return x_min, y_min, x_max, y_max


# 归一化坐标
def normalize_bbox(x_center, y_center, width, height, img_width, img_height):
    x_center = x_center / img_width
    y_center = y_center / img_height
    width = width / img_width
    height = height / img_height
    return x_center, y_center, width, height


def is_parking_violation(detection, category_id, blind_way_mask, blind_way_mask2, img_height, img_width):
    x_center, y_center, width, height = detection
    # 得归一化坐标
    x_center, y_center, width, height = normalize_bbox(x_center, y_center, width, height, img_width, img_height)
    x_min, y_min, x_max, y_max = xywh2xyxy(x_center, y_center, width, height, img_width, img_height)

    # 确保检测框在图像范围内
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(img_width, x_max), min(img_height, y_max)

    # 提取检测框内的盲道分割结果，一个原图的，一个全局图的
    roi_mask = blind_way_mask[y_min:y_max, x_min:x_max]
    roi_mask2 = blind_way_mask2[y_min:y_max, x_min:x_max]

    #     现在盲道分割结果是从原图上进行分割的，我接下来要实现框内盲道像素点数量和全局图框内盲道像素点数量之比,该方法不需要分割其他的像素，如汽车像素,步骤是:
    # 首先应该获得全局图，通过输入或者高斯建模
    # 首先框要映射到原图和全局图的分割结果上,然后从原图分割结果上得到框内盲道像素点数量,
    # 还有从全局图分割结果中,得到全局图框内盲道像素点数量。相除乘100%

    # 统计原图和全局图框内盲道像素点数量
    pixel_count = np.sum(roi_mask == 255)
    pixel_count2 = np.sum(roi_mask2 == 255)
    # 可以理解成计算盲道被遮挡的比例的，也可以理解成增加了多少像素点的占比70,1000
    # 公式一：（全局图框内盲道像素点数量-原图框内盲道像素点数量）/全局图框内盲道像素点数量=(1000-70)/1000=0.93
    # 公式二：1-（原图框内盲道像素点数量/全局图框内盲道像素点数量）=1-（70/1000）=1-0.07=0.93这两个本质是一样的
    if pixel_count2 > 0:
        occupy = 1 - (pixel_count / pixel_count2)
    else:
        occupy = 0
    # # 如果检测框内包含盲道像素点，则判定为违停（这里假设类别号为0表示车辆）
    if occupy >= 0.7:
        return True, (x_min, y_min, x_max, y_max)  # 返回True和违停框坐标
    return False, (x_min, y_min, x_max, y_max)


def main():
    print("请输入背景图,你可以直接输入背景图，或者输入一段视频来生成背景图")
    # 分支1.获取背景图
    image_path = 'background'
    # 分支2.输入视频来生成背景图
    # extract_static_background("car1.avi", output_path="static_background.jpg")
    # 背景图会存放到某一路径下# background生成的背景图

    # 2.语义分割模型对背景图像进行分割，得到盲道的分割结果
    classes = 1  # exclude background
    weights_path = "weights/model_best.pth"
    # 背景图路径
    img_path = image_path
    txt_path = "background/predict.txt"
    save_result = "background/background_mask"  # 背景图分割后存放路径

    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = GRFBUNet(in_channels=3, num_classes=classes + 1, base_c=32)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    total_time = 0
    count = 0
    with open(os.path.join(txt_path), 'r') as f:
        file_name = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
    for file in file_name:
        original_img = Image.open(os.path.join(img_path, file + ".jpg")).convert('RGB')
        count = count + 1
        h = np.array(original_img).shape[0]
        w = np.array(original_img).shape[1]

        data_transform = transforms.Compose([transforms.Resize(565),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])
        img = data_transform(original_img)
        # expand batch dimension

        img = torch.unsqueeze(img, dim=0)

        model.eval()  # Entering Validation Mode
        with torch.no_grad():
            # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            output = model(img.to(device))
            t_end = time_synchronized()
            total_time = total_time + (t_end - t_start)
            print("inference+NMS time: {}".format(t_end - t_start))

            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            prediction = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_LINEAR)
            # Change the pixel value corresponding to the foreground to 255 (white)
            prediction[prediction == 1] = 255
            # Set the pixels in the area of no interest to 0 (black)
            prediction[prediction == 0] = 0
            mask = Image.fromarray(prediction)
            mask = mask.convert("L")
            name = file[-4:]

            if not os.path.exists(save_result):
                os.makedirs(save_result)

            mask.save(os.path.join(save_result, f'background_mask.png'))
    fps = 1 / (total_time / count)
    print("FPS: {}".format(fps))
    # 违规存放的截图
    output_dir = 'results'
    video_detection_and_segmentation(save_result, output_dir)


def segment_image(image, model, device):
    """
    对输入图像进行语义分割。

    参数:
        image: 输入图像（PIL 格式）。
        model: 语义分割模型。
        device: 计算设备（如 'cuda' 或 'cpu'）。

    返回:
        numpy.ndarray: 分割掩码图（单通道，值为 0 或 255）。
    """
    # 图像预处理
    data_transform = transforms.Compose([
        transforms.Resize(565),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=(0.709, 0.381, 0.224), std=(0.127, 0.079, 0.043))  # 归一化
    ])
    image_tensor = data_transform(image).unsqueeze(0).to(device)  # 增加 batch 维度并发送到设备

    # 使用模型进行推理
    with torch.no_grad():
        output = model(image_tensor)  # 前向传播
        prediction = output['out'].argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)  # 获取预测结果
        prediction = cv2.resize(prediction, (image.size[0], image.size[1]), interpolation=cv2.INTER_LINEAR)  # 调整大小
        prediction[prediction == 1] = 255  # 将盲道区域标记为 255
        prediction[prediction == 0] = 0  # 将其他区域标记为 0

    return prediction


# 这边完成了离线部分，接下来实时部分
# 使用YOLO对输入测试视频实时进行目标检测，得到汽车和非机动车的识别框。使用YOLO实时检测，检测到汽车，就把那一帧截下来，进行实时语义分割，把框的坐标对应到这个语义分割图中和背景的语义分割图中，得到对应位置盲道的像素点数量，两个图的像素点数量相除
# 1.函数输入测试的视频，背景图的分割图路径，输出路径，先加载好两个模型，检测和分割模型
# 2.打开检测的视频文件，使用YOLO来检测视频，检测到汽车，就把那一帧截下来，（但是注意，人行道上不会只有一辆车，同一帧可能会有多辆车检测到，也就是多个框，所以同一帧多个框每个框都得对应到背景掩码图上）进行实时语义分割，把框的坐标对应到这个语义分割图中和背景的语义分割图中，得到对应位置盲道的像素点数量，两个图的像素点数量相除
# 输入视频检测
def video_detection_and_segmentation(background_mask_path, output_dir):
    # 加载YOLO模型
    yolo_model = YOLO(model='weights/v11wubest.pt')
    out = 'runs/detect/predict'
    if os.path.exists(out):
        shutil.rmtree(out)
    # 加载语义分割模型
    classes = 1  # exclude background
    weights_path = "weights/model_best.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    segmentation_model = GRFBUNet(in_channels=3, num_classes=classes + 1, base_c=32)
    segmentation_model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    segmentation_model.to(device)
    segmentation_model.eval()

    # 打开视频文件
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return


    # 获取摄像头的基本信息
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera FPS: {fps}")

    # 获取文件夹中的所有文件
    background_mask_files = os.listdir(background_mask_path)
    # 过滤出图片文件（支持 .png 和 .jpg）
    image_files = [f for f in background_mask_files if f.endswith(".png") or f.endswith(".jpg")]
    background_mask_path = os.path.join(background_mask_path, image_files[0])
    # 加载背景图的语义分割结果
    background_mask = Image.open(background_mask_path).convert('L')
    background_mask = np.array(background_mask)
    blind_way_mask2 = cv2.imread(background_mask_path, cv2.IMREAD_GRAYSCALE)
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 初始化视频写入器
    output_video_path = os.path.join(output_dir, "output_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码器
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # 初始化帧计数器
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video processing complete.")
            break

        frame_count += 1
        print(f"Processing frame {frame_count}")

        # 使用YOLO进行目标检测
        results = yolo_model.predict(source=frame, save=False, show=False)
        detection_boxes = results[0].boxes.xywh.cpu().numpy()
        category_ids = results[0].boxes.cls.cpu().numpy()
        count = frame_count
        segimg = frame.copy()
        violation_found = False
        violation_boxes = []
        # 遍历检测结果
        for detection, category_id in zip(detection_boxes, category_ids):
            # if detection['class'] == 'car':  # 假设检测到的类别为'car'
            if count == frame_count:
                # 对当前帧进行语义分割，生成 blind_way_mask
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # 转换为 PIL 格式
                blind_way_mask = segment_image(frame_pil, segmentation_model, device)  # 实时生成分割掩码
                count = count + 1
                # 将掩码图转换为彩色图像
            # 同一帧的长宽不需要变，同一帧的掩码图和背景掩码图也不用变
            violation, violation_box = is_parking_violation(detection, category_id, blind_way_mask, blind_way_mask2,
                                                            frame_height, frame_width)
            if violation:
                violation_found = True
                violation_boxes.append(violation_box)
                # 绘制违停框（红色）
                cv2.rectangle(segimg, violation_box[:2], violation_box[2:], (0, 0, 255), 2)
            else:
                # 绘制正常检测框（绿色）
                cv2.rectangle(segimg, violation_box[:2], violation_box[2:], (0, 255, 0), 2)
        # 如果当前帧有违停，保存违停帧
        if violation_found:
            violation_frame_path = os.path.join(output_dir, f"violation_frame_{frame_count}.jpg")
            cv2.imwrite(violation_frame_path, segimg)
            print(f"Violation found in frame {frame_count}. Saved to {violation_frame_path}")
        # 显示当前帧的检测结果
        cv2.imshow("Video Detection", segimg)
        # 将处理后的帧写入输出视频
        out.write(segimg)
        # 显示绘制后的掩码图


        # 将绘制后的掩码图写入视频
        # 控制播放速度，按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放视频捕获对象并关闭所有窗口
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()






