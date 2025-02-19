import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ['Microsoft YaHei']  # 使用 Microsoft YaHei 字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 定义模态及其解析时延范围（单位：微秒）
modalities = {
    "IP": (4.1, 6.7),
    "ID": (1.8, 5),
    "NDN": (11.1, 20.7),
    "GEO": (5.5, 10.1),
    "FlexIP": (6, 15.4),
    "Custom": (1.3, 5),
}

# 定义带宽范围（从 10Mbps 到 1Gbps）
bandwidths = np.linspace(10, 1000, 50)  # 单位：Mbps

# 定义每种模态的颜色
colors = ['dodgerblue', 'brown', 'green', 'orange', 'purple', 'grey']

# 创建画布
plt.figure(figsize=(14, 6))

# --------------------------
# 折线图
# --------------------------
plt.subplot(1, 2, 1)  # 1行2列，第1个子图

# 折线图标记
markers = ['o', 's', 'D', '^', 'v', '*']
markerssize = [5, 5, 4, 6, 7, 7]

# 为每种模态生成随机时延（在范围内波动）
for i, (modality, (min_delay, max_delay)) in enumerate(modalities.items()):
    delays = np.random.uniform(min_delay, max_delay, len(bandwidths))
    plt.plot(bandwidths, delays, label=modality, linestyle='-', marker=markers[i], markersize=markerssize[i], color = colors[i])

# 设置图表属性
plt.title("解析时延波动", fontsize=14)
plt.xlabel("带宽 (Mbps)", fontsize=12)
plt.ylabel("解析时延 (微秒)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc="upper right", fontsize=10)
plt.xlim(10, 1000)
plt.ylim(0, 22)

# --------------------------
# 柱状图
# --------------------------
plt.subplot(1, 2, 2)  # 1行2列，第2个子图

# 计算每种模态的平均时延
average_delays = [(min_delay + max_delay) / 2 for min_delay, max_delay in modalities.values()]
modality_names = list(modalities.keys())

# 绘制柱状图
bars = plt.bar(modality_names, average_delays, color=colors)

# 为柱状图添加纹理
patterns = ['/', '\\', '|', '-', '+', 'x']
for bar, pattern in zip(bars, patterns):
    bar.set_hatch(pattern)

# 设置图表属性
plt.title("各模态平均解析时延", fontsize=14)
plt.xlabel("模态类型", fontsize=12)
plt.ylabel("平均解析时延 (微秒)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6, axis='y')
plt.ylim(0, 25)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()