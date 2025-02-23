import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体和负号显示
plt.rcParams["font.sans-serif"] = ['Microsoft YaHei']
plt.rcParams["axes.unicode_minus"] = False

# 数据
algorithms = ['Transformer', 'LSTM', 'TCN', 'RNN']
precision = [99.72, 99.60, 99.63, 99.21]  # 精确率
recall = [99.71, 99.59, 99.63, 99.20]     # 召回率
f1_score = [99.71, 99.59, 99.63, 99.20]   # F1分数

# 设置柱状图宽度和位置
bar_width = 0.25
index = np.arange(len(algorithms))

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制柱状图
plt.bar(index, precision, bar_width, label='精确率', color='skyblue')
plt.bar(index + bar_width, recall, bar_width, label='召回率', color='lightgreen')
plt.bar(index + 2 * bar_width, f1_score, bar_width, label='F1分数', color='coral')

# 在柱子上添加数值标签
for i in range(len(algorithms)):
    plt.text(index[i], precision[i] + 0.02, f'{precision[i]}%', ha='center', color='black', fontsize=10)
    plt.text(index[i] + bar_width, recall[i] + 0.02, f'{recall[i]}%', ha='center', color='black', fontsize=10)
    plt.text(index[i] + 2 * bar_width, f1_score[i] + 0.02, f'{f1_score[i]}%', ha='center', color='black', fontsize=10)

# 设置纵轴范围
min_value = min(min(precision), min(recall), min(f1_score)) - 0.5  # 最小值减 0.5
max_value = 99.8
plt.ylim(min_value,max_value)

# 设置图形属性
plt.ylabel('指标', fontsize=12)
plt.title('不同算法模态分类结果对比', fontsize=14)
plt.xticks(index + bar_width, algorithms, fontsize=12)  # 设置横轴标签位置
plt.legend(fontsize=12)

# 显示图形
plt.tight_layout()
plt.show()