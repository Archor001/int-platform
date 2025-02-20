import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# 设置中文字体和负号显示
plt.rcParams["font.sans-serif"] = ['Microsoft YaHei']
plt.rcParams["axes.unicode_minus"] = False

# 数据定义
data1 = [0.9868, 0.9897, 0.9902, 0.9904, 0.9906, 0.9907, 0.9906, 0.9907, 0.9908, 0.9909, 0.9910, 0.9911, 0.9913, 0.9912, 0.9914, 0.9916, 0.9918, 0.9917, 0.9919, 0.9920]
data_len = len(data1) # 获取数据长度，方便后面生成其他数据

# 生成第二组到第四组的波动数据 (单调递增趋势 + 波动)
np.random.seed(42) # 为了结果可复现，设置随机种子

# 生成增量，缩小增量范围，减缓增长速度
increments2 = np.random.uniform(-0.000001, 0.000005, data_len) # 缩小上限为 0.000010
increments3 = np.random.uniform(-0.000001, 0.000005, data_len) # 缩小上限为 0.000010
increments4 = np.random.uniform(-0.000001, 0.000005, data_len) # 缩小上限为 0.000010

# 累积增量，形成单调递增趋势
data2_base = 0.99845
data3_base = 0.99845
data4_base = 0.99847

data2 = data2_base + np.cumsum(increments2)
data3 = data3_base + np.cumsum(increments3)
data4 = data4_base + np.cumsum(increments4)


# x轴数据，假设长度为20
x = range(1, data_len + 1)

# 创建断裂轴图形
fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}) # 移除 sharex=True
fig.subplots_adjust(hspace=0.1, left=0.15, bottom=0.1)  # 调整子图间距和底部边距

# 在ax2（上方子图）绘制数据1
ax2.plot(x, data1, marker='o', linestyle='-', markersize=3, label='RNN', color='blue') # 设置颜色为蓝色

# 在ax1（下方子图）绘制数据2, 3, 4
ax1.plot(x, data2, marker='s', linestyle='-', markersize=3, label='LSTM', color='green') # 设置颜色为绿色
ax1.plot(x, data3, marker='^', linestyle='-', markersize=3, label='TCN', color='red')   # 设置颜色为红色
ax1.plot(x, data4, marker='x', linestyle='-', markersize=3, label='Transformer', color='grey')  # 设置颜色为灰色


# 设置y轴范围 (注意断裂轴需要分别设置)
ax1.set_ylim(min(min(data2), min(data3), min(data4)) - 0.00001, max(max(data2), max(data3), max(data4)) + 0.00001) # 下方子图y轴范围自适应
ax2.set_ylim(0.986, 0.993)    # 上方子图的y轴范围，覆盖data1

# 隐藏上方子图的底部 spines 和下方子图的顶部 spines
ax2.spines['bottom'].set_visible(False)
ax1.spines['top'].set_visible(False)
# ax2.xaxis.tick_top()  #  移除这行！
ax1.xaxis.tick_bottom()

# 移除 *上方* 子图的 x 轴刻度线  ---  正确的做法！
ax1.set_xticks([])

# 添加断裂轴的斜线标识
d = .5  # 设置斜线的长度比例
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle='none', color='k', mec='k', mew=1, clip_on=False)
ax2.plot([0, 1], [0, 0], transform=ax2.transAxes, **kwargs)
ax1.plot([0, 1], [1, 1], transform=ax1.transAxes, **kwargs)

# 添加图例
ax1.legend(loc='upper left')
ax2.legend(loc='lower left')

# 设置x轴标签和y轴标签 (这次设置在 ax1 上，x 轴标签在下方)
ax2.set_xlabel('轮数') #  x 轴标签设置在 ax1，正确的位置！
ax1.set_ylabel('精确率')
ax2.set_ylabel('精确率')

# 设置总标题
fig.suptitle('验证精确率随训练轮数的变化')

# 显示网格 (可选)
ax1.grid(True, linestyle='--', alpha=0.6)
ax2.grid(True, linestyle='--', alpha=0.6)

# 设置 y 轴刻度标签格式为 5 位小数
formatter = ticker.FormatStrFormatter('%.5f')
ax1.yaxis.set_major_formatter(formatter)
ax2.yaxis.set_major_formatter(formatter)

# 显示图形
plt.show()