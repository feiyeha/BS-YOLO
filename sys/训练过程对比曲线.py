import pandas as pd
import matplotlib.pyplot as plt

# 读取两个模型的CSV文件
df_v11n = pd.read_csv("YOLOv11n.csv")     # 原始模型数据
df_improved = pd.read_csv("BS.csv") # 改进模型数据

# 提取数据列（根据实际列名调整！）
# ------------------------------------------------------------
# YOLOv11n 模型数据
epochs_v11n = df_v11n['epoch']
map50_v11n = df_v11n['metrics/mAP50(B)']
map50_95_v11n = df_v11n['metrics/mAP50-95(B)']

# 改进模型数据
epochs_improved = df_improved['epoch']
map50_improved = df_improved['metrics/mAP50(B)']
map50_95_improved = df_improved['metrics/mAP50-95(B)']

# 绘制对比曲线
# ------------------------------------------------------------
plt.figure(figsize=(14, 6))

# 子图1：mAP50对比
plt.subplot(1, 2, 1)
plt.plot(epochs_v11n, map50_v11n, 'b-', label='YOLOv11n', linewidth=2)
plt.plot(epochs_improved, map50_improved, 'r--', label='Improved Model', linewidth=2)
plt.title('mAP50 Comparison', fontsize=12)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('mAP50', fontsize=10)
plt.grid(ls='--', alpha=0.5)
plt.legend()

# 子图2：Recall对比
plt.subplot(1, 2, 2)
plt.plot(epochs_v11n, map50_95_v11n, 'b-', label='YOLOv11n', linewidth=2)
plt.plot(epochs_improved, map50_95_improved, 'r--', label='Improved Model', linewidth=2)
plt.title('mAP50-95 Comparison', fontsize=12)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Recall', fontsize=10)
plt.grid(ls='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig('csv_comparison.png', dpi=300, bbox_inches='tight')
plt.show()