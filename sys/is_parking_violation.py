import numpy as np
def xywh2xyxy(x_center, y_center, width, height, img_width, img_height):
    x_min = int((x_center - width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    x_max = int((x_center + width / 2) * img_width)
    y_max = int((y_center + height / 2) * img_height)
    return x_min, y_min, x_max, y_max
def normalize_bbox(x_center, y_center, width, height, img_width, img_height):
    x_center = x_center / img_width
    y_center = y_center / img_height
    width = width / img_width
    height = height / img_height
    return x_center, y_center, width, height
# 可以理解成计算盲道被遮挡的比例的，也可以理解成增加了多少像素点的占比70,1000 #
# 公式一：（全局图框内盲道像素点数量-原图框内盲道像素点数量）/全局图框内盲道像素点数量=(1000-70)/1000=0.93 # 公式二：1-（原图框内盲道像素点数量/全局图框内盲道像素点数量）=1-（70/1000）=1-0.07=0.93这两个本质是一样的
# 这上面两个公式本质是一样的，但是可以有不同的理解，公式一可以理解成相比原来增加了93%的像素点，增加的量很大，设置70%增加的量超过70%就算违停，
# 然后公式二可以理解成盲道被遮挡的比例，也就是有93%的盲道被遮挡了，还有7%没有被遮挡，设值70%，就是盲道被汽车遮挡的比例超过了70%。遮挡的量大了
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
# 方法一：
# 上面的公式二：理解为盲道被遮挡的比例的，框内盲道像素点数量和空白图的框内盲道像素点数量之比，是的ppt式子I2，但是其实这个是缺陷的，完整应该1-（框内盲道像素点数量和空白图的框内盲道像素点数量之比）他和下面一份确实是同一个，假如89， 全局图90，就说明没有盲道但是89/90=0.98这个概率肯定得1-0.98才行，
# （原图框内盲道像素点数量/全局图框内盲道像素点数量）这个公式代表现在没有被遮挡的像素点占整个框内所有盲道像素点的比例，自然1-（原图框内盲道像素点数量/全局图框内盲道像素点数量）就是被遮挡的那部分像素点占整个框内所有盲道像素点的比例，也就是被遮挡比
# 上面的公式一：框内盲道像素点数量，在空白图的框内盲道像素点数量大幅度增加，是的上面那个就是方法所以也是ppt式子I2  ok

# 框内有多少汽车的像素点在空白图变成了盲道的像素点就是ppt的I1的分子C2c2b，也就是下面第二个的全局图框内的原来是车的像素变成盲道的像素数  ok  遮挡量100个汽车中有20个变成了盲道，，20遮挡量为20


# 汽车的像素点（Car） = 100 个
#
# 盲道的像素点（Sidewalk） = 200 个
#
# 两者重叠的像素点（Overlap） = 50 个
# 方法二
# 车的像素点和空白图中盲道道路的像素点有七八成重叠度，那该如何计算重叠度那，就是计算车的像素点有多少变成了盲道像素点。那其实重叠的部分也就是汽车的像素点在空白图变成了盲道的像素点，也就是这个玩意儿好像也是ppt的I1的分子C2c2b，他是计算重叠度，这个重叠度的计算方式是上面的公式。
# 那些算的遮挡比例，这个算的是，重叠度。车的像素点位置映射到背景图中，他两重叠的部分实际也还是汽车遮挡盲道的部分=汽车的像素点变成盲道像素点的数量（汽车在背景图中会分成盲道和人行道两个像素点）
# 那重叠度的计算就两者交并比的计算，50/[(100+200)-50]=0.25,所以汽车和盲道有25%的重叠度（交并比），这就不是遮挡占用多少盲道遮挡比，而是重叠度
# 方法三：
# 全局图框内的原来是车的像素变成盲道的像素数和原图框内车的像素数比值就是ppt的式子I1  ok其实也是这个车遮挡了多少盲道像素   算的遮挡比例  100个汽车像素点中有20个变成了盲道像素点 20/100=0.2，汽车中有20%部分是遮挡了盲道。汽车是：我的百分之多少（汽车的部分）遮挡了盲道，，汽车覆盖盲道的比例
# 方法四：
# 两者重叠的像素点/背景盲道的像素点数量=20/100=0.2,即框中有20%的盲道被汽车遮挡了，盲道是：我有百分之多少被汽车遮挡了，盲道被遮挡比例


# 这上面都是有全局图的情况
# 这个和1的公式一样如果盲道被遮挡的比例超过 50%，则认为压占盲道
# 第一个也可以理解成这这个被遮挡比例
# 70  1000
# 1000-70=930，930/1000=0.93，相比起原来70增加了93%变成1000，这个是增加了多少像素点百分比
# 1-70/1000=（1000-70）/1000=0.93
# 这个遮挡的比例也就是，两个方法同一个公式
# 两个公式是一样的

def is_parking_violation(detection, category_id, blind_way_mask, blind_way_mask2, img_height, img_width):
    x_center, y_center, width, height = detection
    # 归一化坐标
    x_center, y_center, width, height = normalize_bbox(x_center, y_center, width, height, img_width, img_height)
    x_min, y_min, x_max, y_max = xywh2xyxy(x_center, y_center, width, height, img_width, img_height)

    # 确保检测框在图像范围内
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(img_width, x_max), min(img_height, y_max)

    # 提取检测框内的盲道分割结果
    roi_mask = blind_way_mask[y_min:y_max, x_min:x_max]  # 当前帧的盲道掩码
    roi_mask2 = blind_way_mask2[y_min:y_max, x_min:x_max]  # 背景图的盲道掩码

    # 获取当前帧检测框内的汽车像素点
    car_mask = (roi_mask == 0)  # 假设汽车像素点值为 0
    car_pixel_count = np.sum(car_mask)  # 原图框内车的像素数 C1_car

    # 获取背景图中对应位置的盲道像素点
    background_blind_pixels = roi_mask2[car_mask]  # 背景图中对应汽车像素点的盲道像素值
    c2_c2b = np.sum(background_blind_pixels == 255)  # 背景图中原来是车的像素变成盲道的像素数 C2_c2b

    # 方法 1：框内有多少汽车的像素点在背景图变成了盲道的像素点
    c2_c2b_threshold = 70  # 阈值，可以根据实际情况调整
    is_method1_violation = c2_c2b > c2_c2b_threshold

    # 方法 2：背景图框内的原来是车的像素变成盲道的像素数和原图框内车的像素数比值
    if car_pixel_count > 0:
        ratio = c2_c2b / car_pixel_count  # 计算比值
    else:
        ratio = 0
    ratio_threshold = 0.7  # 阈值，例如 70%
    is_method2_violation = ratio > ratio_threshold

    # 综合判断违停
    if is_method1_violation or is_method2_violation:
        return True, (x_min, y_min, x_max, y_max)  # 返回True和违停框坐标
    return False, (x_min, y_min, x_max, y_max)