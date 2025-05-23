from collections import defaultdict
import time
def calculate_iou(box1, box2):
    """
    计算两个框的 IoU。

    参数:
        box1: [x_min, y_min, x_max, y_max]
        box2: [x_min, y_min, x_max, y_max]

    返回:
        iou: 交并比
    """
    # 计算交集区域的坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集区域面积
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算两个框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集区域面积
    union_area = box1_area + box2_area - inter_area

    # 计算 IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou
class VehicleTimer:
    def __init__(self, violation_threshold=10.0, iou_threshold=0.7):
        self.timers = defaultdict(lambda: {
            "start_time": None,  # 违停开始时间
            "initial_box": None,  # 初始检测框
            "current_box": None,  # 当前检测框
        })
        self.violation_threshold = violation_threshold  # 违停时间阈值
        self.iou_threshold = iou_threshold  # IoU 阈值

    def update(self, track_id, current_box):
        """
        更新计时器。

        参数:
            track_id: 车辆的唯一ID。
            current_box: 当前检测框 [x_min, y_min, x_max, y_max]。

        返回:
            elapsed_time: 当前计时器的时间（秒）。
            is_violation: 是否判定为违停。
        """
        if self.timers[track_id]["start_time"] is None:
            # 首次判定为违停，记录开始时间和初始检测框
            self.timers[track_id]["start_time"] = time.time()
            self.timers[track_id]["initial_box"] = current_box
            self.timers[track_id]["current_box"] = current_box
        else:
            # 更新当前检测框
            self.timers[track_id]["current_box"] = current_box

            # 计算当前检测框与初始检测框的 IoU
            iou = calculate_iou(self.timers[track_id]["initial_box"], current_box)
            if iou < self.iou_threshold:
                # 如果车辆移动显著，重置计时器
                self.timers[track_id]["start_time"] = time.time()
                self.timers[track_id]["initial_box"] = current_box

            # 检查是否超过违停时间阈值
            elapsed_time = time.time() - self.timers[track_id]["start_time"]
            if elapsed_time >= self.violation_threshold:
                return elapsed_time, True

        return 0, False

    def reset(self, track_id):
        """
        重置计时器。
        """
        self.timers[track_id]["start_time"] = None
        self.timers[track_id]["initial_box"] = None
        self.timers[track_id]["current_box"] = None