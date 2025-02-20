import numpy as np
import matplotlib.pyplot as plt

# 分级消息队列优化版

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
    noise = np.random.normal(0, noise_level * ideal_y, len(x)) # Noise scaled by *each* ideal_y
    return noise

# 定义六条曲线
curves = [
    {"name": "容器内通信(拟合)", "points": [(0, 0), (105, 0), (1000, 80)], "type": "ideal"},
    {"name": "跨容器通信(拟合)", "points": [(0, 0), (95, 0), (1000, 130)], "type": "ideal"},
    {"name": "跨域通信(拟合)", "points": [(0, 0), (85, 0), (1000, 150)], "type": "ideal"},
    {"name": "容器内通信", "points": [(0, 0), (100, 0), (1000, 80)], "type": "noisy"},
    {"name": "跨容器通信", "points": [(0, 0), (88, 0), (1000, 130)], "type": "noisy"},
    {"name": "跨域通信", "points": [(0, 0), (80, 0), (1000, 150)], "type": "noisy"}
]

colors = ['blue', 'green', 'red', 'blue', 'green', 'red']
markers = ['o', 's', '^', None, None, None]
noise_levels = [0, 0, 0, 0.05, 0.05, 0.05] # Increased noise levels for better visualization
line_styles = ['--', '--', '--', '-', '-', '-']
ideal_curve_color = 'lightgrey'

x = np.linspace(0, 1000, 50)

plt.figure(figsize=(12, 8))

start_points_x = [100, 88, 80] # User provided start points x-coordinates for noisy curves
text_y_offsets = [-3, 20, -3] # Vertical offsets for text, now uniform, adjust as needed
text_horizontal_alignments = ['left', 'center', 'right']
text_vertical_alignments = ['top', 'bottom', 'top'] # Define vertical alignments for each text

for idx, curve in enumerate(curves):
    if curve["type"] == "ideal":
        a, b, c = calculate_ideal_coefficients(curve["points"])
        y_ideal = quadratic_function(x, a, b, c) # Calculate ideal y for all x
        y = y_ideal # Ideal curve has no noise
        curve_color = ideal_curve_color
        current_marker = markers[idx]
        current_linestyle = line_styles[idx]
        marker_color = colors[idx] # 设置理想曲线 marker 颜色为对应的 colors 列表颜色
    else:
        a, b, c = calculate_ideal_coefficients(curve["points"]) # Use ideal coefficients as base
        y_ideal = quadratic_function(x, a, b, c) # Calculate ideal y for all x
        noise = generate_point_noise(x, y_ideal, noise_levels[idx]) # Generate point-by-point noise
        y = y_ideal + noise # Add noise to the ideal y values
        curve_color = colors[idx]
        current_marker = markers[idx]
        current_linestyle = line_styles[idx]
        marker_color = None

    # Modified plt.plot for ideal curves to have approximately 50 markers
    if curve["type"] == "ideal":
        plt.plot(x, y, label=curve["name"], color=curve_color, linestyle=current_linestyle, marker=current_marker, markevery=4, markersize=6, markerfacecolor=marker_color, markeredgecolor='black', markeredgewidth=0.5) # Reduced markevery to ~40 for ~50 markers
    else:
        plt.plot(x, y, label=curve["name"], color=curve_color, linestyle=current_linestyle, marker=current_marker, markevery=1000, markersize=6, markerfacecolor=marker_color, markeredgecolor=marker_color)

# Annotate the start points for noisy curves
annotation_y_start = 0 # Start y-coordinate for vertical lines
annotation_y_end = 20  # End y-coordinate for vertical lines, control line height

for i in range(3): # For the three noisy curves
    plt.plot([start_points_x[i], start_points_x[i]], [annotation_y_start, annotation_y_end], color=colors[i+3], linestyle='--', linewidth=0.8) # Vertical line with controlled height
    plt.text(start_points_x[i], text_y_offsets[i], f'x={start_points_x[i]}', color=colors[i+3], fontsize=11, ha=text_horizontal_alignments[i], va=text_vertical_alignments[i]) # Text label with staggered vertical alignment

# 添加标题和标签
plt.title('分级消息队列时延折线图', fontsize=16, pad=15)
plt.xlabel('重放速率(Mbps)', fontsize=12)
plt.ylabel('消息队列时延 (ms)', fontsize=12)
plt.xlim(0, 1000)
plt.ylim(0, 250)
plt.grid(True)
plt.legend() # Let matplotlib place the legend automatically
plt.show()