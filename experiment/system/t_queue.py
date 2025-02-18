import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

plt.rcParams["font.sans-serif"] = ['Microsoft YaHei']  # 使用 Microsoft YaHei 字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 1. 定义关键数据点
points_x = np.array([0, 80, 180, 1000])  # 带宽 (Mbps)
points_y = np.array([0, 4.8, 2.2, 122])   # 时延 (ms)

# 2. 使用三次样条插值创建平滑曲线
# CubicSpline 可以保证曲线在节点处一阶和二阶导数连续，使曲线更平滑
cs = CubicSpline(points_x, points_y)

# 3. 生成更密集的 x 值用于绘制平滑曲线
x_smooth = np.linspace(0, 1000, 500) # 在 0-1000 Mbps 范围内生成 500 个点
y_smooth = cs(x_smooth)

# 4. 确保在 x=80 附近是极大值，x=180 附近是极小值
# 三次样条插值会自动尝试拟合数据，但可能不会精确地在 80 和 180 处出现绝对的极大值和极小值。
# 为了更贴近要求，我们可以手动调整曲线形状。
# 由于三次样条已经大致符合，我们这里先不进行复杂的强制极值点调整，
# 如果需要更精确的极值点控制，可能需要更复杂的函数拟合方法。
# 目前的 CubicSpline 通常能较好地反映趋势。

# 5. 绘制曲线和关键点
plt.figure(figsize=(10, 6)) # 设置图像大小，可以调整figsize参数
plt.plot(x_smooth, y_smooth, label='时延曲线', color='blue') # 绘制平滑曲线
plt.scatter(points_x, points_y, color='red', label='关键数据点', s=50) # 标记关键点

# 6. 添加标签、标题和网格
plt.xlabel('带宽 (Mbps)')
plt.ylabel('时延 (ms)')
plt.title('带宽与时延关系曲线')
plt.grid(True, linestyle='--', alpha=0.7) # 添加网格，虚线，半透明

# 7. 标注极值点 (近似标注，三次样条可能不会精确在80和180取到极值，但会趋势接近)
plt.annotate('极大值点 (近似)', xy=(80, 4.8), xytext=(100, 6),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )
plt.annotate('极小值点 (近似)', xy=(180, 2.2), xytext=(200, 1),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )

# 8. 限制坐标轴范围，并显示图例
plt.xlim(0, 1000)
plt.ylim(0, max(points_y) * 1.2) # Y轴上限稍微留出空间
plt.legend()

# 9. 显示图像
plt.show()

# 10. 保存图像 (如果需要保存到文件)
# plt.savefig('delay_bandwidth_curve.png', dpi=300, bbox_inches='tight')