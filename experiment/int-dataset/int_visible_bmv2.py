import pandas as pd
import matplotlib.pyplot as plt

# 读取生成的 CSV 文件
filename = "int_data_bmv2_all.csv"
data = pd.read_csv(filename)

# 创建画布和子图
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# 绘制端到端时延（e2e Delay）
ax1.plot(data["Packet ID"], data["e2e Delay (ms)"], label="e2e Delay (ms)", color="blue")
ax1.set_ylabel("e2e Delay (ms)")
ax1.set_title("End-to-End Delay, Queue Occupancy, and IAT with Anomaly Regions (Hop:4, Bandwith: 100Mbps)")
ax1.legend(loc="upper left")
ax1.grid(True)

# 绘制队列深度（Queue Occupancy）
ax2.plot(data["Packet ID"], data["Queue Occupancy"], label="Queue Occupancy", color="green")
ax2.set_ylabel("Queue Occupancy")
ax2.legend(loc="upper left")
ax2.grid(True)

# 绘制数据包到达间隔（IAT）
ax3.plot(data["Packet ID"], data["IAT (us)"], label="IAT (us)", color="orange")
ax3.set_xlabel("Packet ID")
ax3.set_ylabel("IAT (us)")
ax3.legend(loc="upper left")
ax3.grid(True)

# 标注异常流量区间
anomaly_regions = data[data["Is Anomaly"] == 1]
for _, row in anomaly_regions.iterrows():
    ax1.axvspan(row["Packet ID"], row["Packet ID"] + 1, color="grey", alpha=0.3)
    ax2.axvspan(row["Packet ID"], row["Packet ID"] + 1, color="grey", alpha=0.3)
    ax3.axvspan(row["Packet ID"], row["Packet ID"] + 1, color="grey", alpha=0.3)

# 调整布局
plt.tight_layout()

# 显示图像
plt.show()