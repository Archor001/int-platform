import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ['Microsoft YaHei']
plt.rcParams["axes.unicode_minus"] = False

# 1. 准备 x 轴数据 (带宽 Mbps)
x = np.linspace(0, 1000, 50)  # 在 0 到 1000 之间生成 100 个均匀分布的点，作为带宽数据

# 2. 准备 y 轴数据 (时延 毫秒) - 模拟波动
np.random.seed(42)  # 设置随机种子，使结果可复现

# 容器内通信 (4ms 附近波动)
y_container_internal = 4 + np.random.normal(0, 0.3, len(x))  # 平均值 4ms，标准差 0.3ms 的正态分布噪声

# 跨容器通信 (10ms 附近波动)
y_cross_container = 10 + np.random.normal(0, 0.8, len(x)) # 平均值 10ms，标准差 0.8ms 的正态分布噪声

# 跨域通信 (12ms 附近波动)
y_cross_domain = 12 + np.random.normal(0, 1.0, len(x))   # 平均值 12ms，标准差 1.0ms 的正态分布噪声

# 3. 绘制折线图
plt.figure(figsize=(10, 6))  # 设置图像大小 (可选)

plt.plot(x, y_container_internal, label='容器内通信', color='blue', linestyle='-', marker='o', markersize=4)
plt.plot(x, y_cross_container, label='跨容器通信', color='green', linestyle='-', marker='s', markersize=4)
plt.plot(x, y_cross_domain, label='跨域通信', color='red', linestyle='-', marker='^', markersize=4)

# 4. 添加图例、标签和标题
plt.xlabel('重放速率 (Mbps)')
plt.ylabel('路由计算时延 (毫秒)')
plt.title('路由计算时延折线图')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# 5. 设置 x 和 y 轴范围 (可选，根据需要调整)
plt.xlim(0, 1000)
plt.ylim(0, 15)  # 时延范围可以根据实际波动情况调整，这里设置为 0-15ms

# 6. 显示图像
plt.show()