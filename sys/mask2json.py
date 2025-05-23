import cv2
import json
import os
import numpy as np
from pathlib import Path
import os
import cv2
import json
import numpy as np
from glob import glob


def png_mask_to_json(mask_folder, output_folder, image_ext=".jpg"):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有PNG文件
    mask_files = glob(os.path.join(mask_folder, "*.png"))

    for mask_path in mask_files:
        # 读取掩码图像
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        height, width = mask.shape

        # 准备JSON数据结构
        json_data = {
            "version": "2.5.4",
            "flags": {},
            "shapes": [],
            "imagePath": os.path.basename(mask_path).replace(".png", image_ext),
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
            "description": ""
        }

        # 查找白色区域（255）的轮廓
        contours, _ = cv2.findContours(
            (mask == 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # 为每个轮廓创建多边形
        for contour in contours:
            # 简化轮廓（epsilon值可根据需要调整）
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # 转换为点坐标列表
            points = [[float(point[0][0]), float(point[0][1])]
                      for point in approx]

            # 跳过无效多边形
            if len(points) < 3:
                continue

            # 添加形状到JSON
            json_data["shapes"].append({
                "kie_linking": [],
                "label": "blind sidewalk",
                "score": None,
                "points": points,
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "polygon",
                "flags": {},
                "attributes": {}
            })

        # 保存JSON文件
        output_path = os.path.join(
            output_folder,
            os.path.basename(mask_path).replace(".png", ".json")
        )
        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=2)


# 使用示例
png_mask_to_json(
    mask_folder="labels",
    output_folder="Part08",
    image_ext=".jpg"  # 根据实际图片格式修改
)
