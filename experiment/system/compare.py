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

colors = ['blue', 'green', 'red']  # Colors for communication scenarios, not directly used now
markers_noisy = ['o', 's', '^']  # Markers for noisy curves
markers_for_selected = ['h', 'p', 'd'] # Markers for selected curves: Cross-container, Weighted average, New Fitted (Noisy)
noise_levels = [0, 0, 0, 0.03, 0.03, 0.03] # Increased noise levels for better visualization
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

# 计算加权平均 "首台交换机时延" (仅使用噪声曲线)
weights = [0.125, 0.125, 0.75]  # 容器内, 跨容器, 跨域
weighted_y_first_switch_noisy = np.average(y_first_switch_curves_noisy, axis=0, weights=weights)

# ---  新的拟合曲线 (带噪声) ---
new_curve_points = [(0, 9), (350, 23), (1000, 195)] # 调整拟合点以强制单调递增
a_new, b_new, c_new = calculate_ideal_coefficients(new_curve_points)
y_new_fitted_ideal = quadratic_function(x, a_new, b_new, c_new) # 先计算理想曲线
noise_level_new_curve = 0.02 # 可以调整新的曲线的噪声水平
noise_new_curve = generate_point_noise(x, y_new_fitted_ideal, noise_level_new_curve)
y_new_fitted_noisy = y_new_fitted_ideal + noise_new_curve # 添加噪声
# --- 新的拟合曲线 (带噪声) 结束 ---


# 创建画布和子图 (只有一个子图 ax1)
fig, ax1 = plt.subplots(figsize=(12, 6))  # Adjusted figure size for single plot

# 绘制 "跨容器通信" 噪声曲线
cross_container_index = 1 # Index for "跨容器通信" in noisy curves list
ax1.plot(x, y_first_switch_curves_noisy[cross_container_index], label=curves_data[cross_container_index + 3]['name'], color='darkcyan', linestyle=line_styles_noisy[cross_container_index], marker=markers_for_selected[0], markevery=4, markersize=6)

# 绘制加权平均曲线
ax1.plot(x, weighted_y_first_switch_noisy, label='本架构多模态', color='indigo', linestyle='-', linewidth=2, marker=markers_for_selected[1], markevery=4, markersize=6)

# --- 绘制新的拟合曲线 (带噪声) ---
ax1.plot(x, y_new_fitted_noisy, label='传统架构单模态', color='coral', linestyle='-', marker=markers_for_selected[2], markevery=4, markersize=6)
# --- 新的拟合曲线 (带噪声) 绘制结束 ---


# 设置子图属性 (Y轴范围，刻度，标签，标题，图例，网格...)
ax1.set_ylim(8, max(np.max(y_first_switch_curves_noisy[cross_container_index]), np.max(weighted_y_first_switch_noisy), np.max(y_new_fitted_noisy)) * 1.2)
ax1.set_yticks(np.arange(10, max(np.max(y_first_switch_curves_noisy[cross_container_index]), np.max(weighted_y_first_switch_noisy), np.max(y_new_fitted_noisy)) * 1.2, 30))

# 设置横轴刻度值
xticks = np.arange(0, 1001, 100)  # 从0到1000，每隔100一个刻度
ax1.set_xticks(xticks)  # 设置刻度位置
ax1.set_xticklabels([str(int(x)) for x in xticks], fontsize=12)  # 设置刻度标签

# 设置横轴标签
ax1.set_xlabel('重放速率 (Mbps)', fontsize=14)

# 设置纵轴标签
ax1.set_ylabel('通信时延 (ms)', fontsize=14)

# 设置标题
plt.title('通信时延与重放速率关系图 (本架构多模态 & 本架构单模态 & 传统架构单模态)', fontsize=18, y=1.02) # Adjusted title

# 添加网格
ax1.grid(True, linestyle='--', alpha=0.6)

# 添加图例
legend_lines = [Line2D([0], [0], color='darkcyan', marker=markers_for_selected[0], linestyle='-', markersize=6, label='本架构单模态'),
                 Line2D([0], [0], color='indigo', marker=markers_for_selected[1], linestyle='-', linewidth=2, markersize=6, label='本架构多模态'),
                 Line2D([0], [0], color='coral', marker=markers_for_selected[2], linestyle='-', markersize=6, label='传统架构单模态')]
ax1.legend(handles=legend_lines, loc='upper right', fontsize=9, ncol=1) # Adjusted legend

# 显示图表
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()