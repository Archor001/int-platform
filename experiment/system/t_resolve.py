import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ['Microsoft YaHei']
plt.rcParams["axes.unicode_minus"] = False

# 定义模态及其解析时延范围（单位：微秒）
modalities = {
    "IP": (4.1, 6.7),
    "ID": (1.8, 5),
    "NDN": (11.1, 20.7),
    "GEO": (5.5, 10.1),
    "FlexIP": (6, 15.4),
    "Custom": (1.3, 5),
}

# 第二项数据：流表下发时延平均值（单位：微秒）
flow_rule_delays_avg = [19.5, 16.3, 29.0, 28.3, 25.1, 15.4]

# 第三项数据：端到端通信非首台交换机时延平均值（单位：微秒）
transit_switch_hop_latency_avg = [10.7, 6.4, 25.5, 18.2, 13.7, 8.6]

# 定义颜色
parse_delay_color = 'purple'      # 模态解析时延颜色
flow_delay_color = 'grey'       # 流表下发时延颜色
switch_hop_latency_color = 'orange' # 交换机时延颜色

bar_width = 0.25  # 柱状图宽度
index = np.arange(len(modalities)) # 柱状图横坐标

# 计算每种模态的平均解析时延
average_delays = [(min_delay + max_delay) / 2 for min_delay, max_delay in modalities.values()]
modality_names = list(modalities.keys())

# 创建画布
plt.figure(figsize=(10, 6)) # 调整画布大小，适应单图

# --------------------------
# 柱状图
# --------------------------

# 绘制第一组柱状图：解析时延 (shifted to the left)
bars1 = plt.bar(index - bar_width, average_delays, bar_width, label='模态解析时延', color=parse_delay_color)

# 绘制第二组柱状图：流表下发时延 (centered)
bars2 = plt.bar(index , flow_rule_delays_avg, bar_width, label='流表下发时延', color=flow_delay_color)

# 绘制第三组柱状图：端到端通信非首台交换机时延 (shifted to the right)
bars3 = plt.bar(index + bar_width, transit_switch_hop_latency_avg, bar_width, label='交换机（非首台）时延', color=switch_hop_latency_color)


# 添加柱状图数值标签
def add_value_labels(bars, color):
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.1f}', # 格式化数值，保留一位小数
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom',
                     color=color) # 设置文字颜色与柱子颜色一致

add_value_labels(bars1, parse_delay_color)
add_value_labels(bars2, flow_delay_color)
add_value_labels(bars3, switch_hop_latency_color)


# 设置横坐标刻度标签位置居中 - now at the center of the groups
plt.xticks(index, modality_names)

# 设置图表属性
plt.title("不同模态的解析时延、流表下发时延和交换机（非首台）时延对比", fontsize=14) # 修改标题
plt.xlabel("模态类型", fontsize=12)
plt.ylabel("平均时延 (微秒)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6, axis='y')
plt.ylim(0, 35) # 调整 Y 轴上限以适应新的数据范围
plt.legend(loc='upper left', fontsize=10) # 添加图例，并调整位置

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()