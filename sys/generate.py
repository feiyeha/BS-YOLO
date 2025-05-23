import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#
# # 定义类别标签
# labels = ['complete', 'incomplete', 'background']
#
# # 生成一个随机的混淆矩阵，对角线上的数值在0.9左右
# conf_matrix = np.random.rand(3, 3) * 0.1
# np.fill_diagonal(conf_matrix, 0.9)
#
# # 归一化混淆矩阵，使得每一列的和为1
# conf_matrix = conf_matrix / conf_matrix.sum(axis=0, keepdims=True)
#
# # 使用seaborn绘制混淆矩阵
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues",
#             xticklabels=labels, yticklabels=labels)
#
# # 设置轴标签
# plt.xlabel('True', fontsize=12)
# plt.ylabel('Predicted', fontsize=12)
# plt.title('Confusion Matrix (Column Normalized)', fontsize=14)
#
# # 保存图像到本地
# output_path = 'confusion_matrix.png'  # 保存路径和文件名
# plt.savefig(output_path, dpi=300, bbox_inches='tight')  # dpi 设置分辨率，bbox_inches 确保图像完整保存
#
# # 显示图像（可选）
# plt.show()
#
# print(f"混淆矩阵已保存到: {output_path}")


import matplotlib.pyplot as plt

# 模型名称和推理时间（单位：毫秒）
models = ['YOLOv5', 'Faster_YOLOv5']
inference_times = [54, 32]  # 推理时间

# 绘制水平柱状图
plt.figure(figsize=(8, 4))
bars = plt.barh(models, inference_times, color=['red', 'green'])

# 添加标题和标签
plt.title('Inference Time Comparison')
plt.xlabel('Inference Time (ms)')
plt.ylabel('Model')

# 在柱形右侧标注数值
for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width} ms', va='center')

# 保存图像到本地
output_path = 'inference_time_comparison.png'  # 保存路径和文件名
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # dpi 设置分辨率，bbox_inches 确保图像完整保存

# 显示图像（可选）
plt.show()

print(f"图像已保存到: {output_path}")
