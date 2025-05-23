
import cv2
import numpy as np

#加了某一帧中没有运动物体出现，那么这一帧可以直接作为高质量的静态背景图像
def extract_static_background(video_path, output_path="static_background.jpg", motion_threshold=1000):
    """
    提取静态背景图像
    :param video_path: 视频文件路径
    :param output_path: 静态背景保存路径
    :param motion_threshold: 运动物体像素数量阈值，低于该阈值则认为没有运动物体
    """
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 初始化 GMM 背景减除器
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    # 用于存储上一帧
    last_frame = None

    while True:
        # 读取当前帧
        ret, frame = cap.read()
        if not ret:
            break

        # 使用 GMM 更新背景模型
        fg_mask_gmm = bg_subtractor.apply(frame)

        # 计算帧间差分
        if last_frame is not None:
            fg_mask_diff = cv2.absdiff(last_frame, frame)
            fg_mask_diff = cv2.cvtColor(fg_mask_diff, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
            _, fg_mask_diff = cv2.threshold(fg_mask_diff, 30, 255, cv2.THRESH_BINARY)

            # 结合 GMM 和帧间差分的结果
            combined_mask = cv2.bitwise_or(fg_mask_gmm, fg_mask_diff)
        else:
            combined_mask = fg_mask_gmm

        # 对前景掩码进行形态学操作（消除噪声）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # 计算运动物体的像素数量
        motion_pixel_count = cv2.countNonZero(combined_mask)

        # 如果没有检测到运动物体，保存当前帧为背景图像
        if motion_pixel_count < motion_threshold:
            cv2.imwrite(output_path, frame)
            print(f"Static background saved to {output_path} (no motion detected).")
            break

        # 更新上一帧
        last_frame = frame

    # 如果循环结束仍未找到干净的背景帧，使用 GMM 的背景图像
    if motion_pixel_count >= motion_threshold:
        print("No frame without motion detected. Using GMM background image.")
        background = bg_subtractor.getBackgroundImage()
        cv2.imwrite(output_path, background)
        print(f"Static background saved to {output_path} (GMM background).")

# 调用函数
extract_static_background("generate background.mp4", output_path="background/static_background.jpg")
