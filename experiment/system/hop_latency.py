import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# 设置中文字体和负号显示
plt.rcParams["font.sans-serif"] = ['Microsoft YaHei']
plt.rcParams["axes.unicode_minus"] = False

# 定义二次函数
def quadratic_function(x, a, b, c):
    y = a * x**2 + b * x + c
    y[y < 0] = 0
    return y

# 定义计算理想曲线系数的函数 (无噪声)
def calculate_ideal_coefficients(points):
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    A = np.array([[x1**2, x1, 1], [x2**2, x2, 1], [x3**2, x3, 1]])
    B = np.array([y1, y2, y3])
    a, b, c = np.linalg.solve(A, B)
    return a, b, c

# 定义函数生成点对点噪声
def generate_point_noise(x, ideal_y, noise_level):
    """Generates point-by-point noise scaled to the ideal y-value at each x."""
    noise = np.random.normal(0, noise_level * ideal_y, len(x))  # Noise scaled by *each* ideal_y
    return noise

# 定义六条曲线
curves_data = [
    {"name": "容器内通信(拟合)", "points": [(0, 0), (105, 0), (1000, 80)], "type": "ideal"},
    {"name": "跨容器通信(拟合)", "points": [(0, 0), (95, 0), (1000, 130)], "type": "ideal"},
    {"name": "跨域通信(拟合)", "points": [(0, 0), (85, 0), (1000, 150)], "type": "ideal"},
    {"name": "容器内通信", "points": [(0, 0), (100, 0), (1000, 80)], "type": "noisy"},
    {"name": "跨容器通信", "points": [(0, 0), (88, 0), (1000, 130)], "type": "noisy"},
    {"name": "跨域通信", "points": [(0, 0), (80, 0), (1000, 150)], "type": "noisy"}
]

colors = ['blue', 'green', 'red']  # Colors for communication scenarios
markers_noisy = ['o', 's', '^']  # Markers for noisy curves
markers_ideal = ['o', 's', '^']  # Markers for ideal curves - can be same or different
noise_levels = [0, 0, 0, 0.05, 0.05, 0.05]  # Increased noise levels for better visualization
line_styles_noisy = ['-', '-', '-']
line_styles_ideal = ['--', '--', '--']
ideal_curve_color = 'lightgrey'

x = np.linspace(0, 1000, 50)

# 计算所有曲线的 y 值
curve_y_values = []
for idx, curve in enumerate(curves_data):
    if curve["type"] == "ideal":
        a, b, c = calculate_ideal_coefficients(curve["points"])
        y_ideal = quadratic_function(x, a, b, c)
        curve_y_values.append(y_ideal)
    else:
        a, b, c = calculate_ideal_coefficients(curve["points"])
        y_ideal = quadratic_function(x, a, b, c)
        noise = generate_point_noise(x, y_ideal, noise_levels[idx])
        y_noisy = y_ideal + noise
        curve_y_values.append(y_noisy)

# 计算 "首台交换机时延" 和 "后续交换机时延" for each scenario
y_first_switch_curves_noisy = []  # Noisy first switch delays
y_first_switch_curves_ideal = []  # Ideal (fitted) first switch delays
y_subsequent_switch_curves = []

for i in range(3):  # For Container Internal, Cross-Container, Cross-Domain
    y_first_switch_noisy = curve_y_values[i + 3] + 10  # Use noisy curves for noisy first switch delay
    y_first_switch_ideal = curve_y_values[i] + 10  # Use ideal curves for ideal first switch delay
    y_subsequent_switch = 0.01 + np.random.normal(0, 0.002, len(x))  # Subsequent switch delay is similar for all

    y_first_switch_curves_noisy.append(y_first_switch_noisy)
    y_first_switch_curves_ideal.append(y_first_switch_ideal)
    y_subsequent_switch_curves.append(y_subsequent_switch)

# 创建画布和子图 (断裂轴)
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

# 绘制 "首台交换机时延" 曲线 (上方子图)
for i in range(3):  # 绘制三种通信场景的 "首台交换机时延"
    ax1.plot(x, y_first_switch_curves_noisy[i], label=curves_data[i + 3]['name'], color=colors[i], linestyle=line_styles_noisy[i])  # Scenario label only
    # 绘制拟合曲线
    ax1.plot(x, y_first_switch_curves_ideal[i], label=curves_data[i]['name'], color=ideal_curve_color, linestyle=line_styles_ideal[i], marker=markers_ideal[i], markevery=4, markersize=6, markerfacecolor=colors[i], markeredgecolor='black', markeredgewidth=0.5)  # Scenario label only

# 绘制 "后续交换机时延" 曲线 (下方子图)
for i in range(3):  # 绘制三种通信场景的 "后续交换机时延"，颜色区分场景
    ax2.plot(x, y_subsequent_switch_curves[i], label=curves_data[i + 3]['name'], color=colors[i], linestyle=line_styles_noisy[i], marker=markers_noisy[i], markevery=4, markersize=6)  # Scenario label only

# 设置子图属性 (Y轴范围，刻度，标签，标题，图例，网格...)
ax1.set_ylim(8, max(np.max(y_first_switch_curves_noisy[0]), np.max(y_first_switch_curves_noisy[1]), np.max(y_first_switch_curves_noisy[2])) * 1.2)
ax1.set_yticks(np.arange(10, max(np.max(y_first_switch_curves_noisy[0]), np.max(y_first_switch_curves_noisy[1]), np.max(y_first_switch_curves_noisy[2])) * 1.2, 30))
ax2.set_ylim(0, 0.02)
ax2.set_yticks(np.arange(0, 0.021, 0.005))

# 设置横轴刻度值
xticks = np.arange(0, 1001, 100)  # 从0到1000，每隔100一个刻度
ax2.set_xticks(xticks)  # 设置刻度位置
ax2.set_xticklabels([str(int(x)) for x in xticks], fontsize=12)  # 设置刻度标签

# 隐藏ax1的横轴刻度
ax1.set_xticks([])

# 设置横轴标签
ax2.set_xlabel('带宽 (Mbps)', fontsize=14)

# 设置纵轴标签
fig.text(0.06, 0.5, '交换机时延 (ms)', va='center', rotation='vertical', fontsize=14)

# 设置标题
plt.suptitle('交换机时延与带宽关系', fontsize=18, y=0.93)

# 添加网格
ax1.grid(True, linestyle='--', alpha=0.6)
ax2.grid(True, linestyle='--', alpha=0.6)

# 添加图例
legend_colors = [Line2D([0], [0], color=colors[0], label='容器内通信'),
                 Line2D([0], [0], color=colors[1], label='跨容器通信'),
                 Line2D([0], [0], color=colors[2], label='跨域通信')]
ax1.legend(handles=legend_colors, loc='upper right', fontsize=9, ncol=3)
ax2.legend(handles=legend_colors, loc='upper right', fontsize=9, ncol=3)

# 断裂轴视觉效果
d = .015
kwargs = dict(marker=[(-1, -d), (1, d)], linestyle='none',
              markersize=12, markeredgewidth=1, color='k',
              clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

# 显示图表
plt.show()