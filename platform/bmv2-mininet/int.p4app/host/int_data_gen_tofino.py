import random
import csv
import numpy as np
import math

# 定义模态类型及其比例因子
MODAL_TYPES = {
    "NDN": 1.8,
    "GEO": 1.4,
    "FlexIP": 1.2,
    "IPv4": 1.0,
    "ID": 0.8,
    "MF": 0.6
}

# 定义跳数
NUM_HOPS = 4  # 直线拓扑中的跳数


def fast_approaching(x, k=5):
    """
    自定义函数：y = (1 - exp(-k * x)) / (1 - exp(-k))

    参数:
        x (float): 输入值，范围 [0, 1]。
        k (float): 控制函数增长速度的参数，默认值为 5。

    返回:
        y (float): 输出值，范围 [0, 1]。
    """
    return (1 - np.exp(-k * x)) / (1 - np.exp(-k))

# 定义生成数据的函数
def generate_network_data(num_samples, anomaly_prob=0.01):
    data = []
    is_burst = False  # 是否处于微突发流量状态
    burst_start = 0  # 微突发流量的起始位置
    burst_end = 0  # 微突发流量的结束位置
    burst_duration = 0  # 微突发流量的持续时间
    burst_peak = 0  # 指标达到峰值的时刻
    burst_magnification = random.uniform(5, 13)  # 突发流量的放大倍数
    base_iat = 100  # 正常情况下的 IAT 基础值（微秒）
    min_iat = 20    # 突发期间 IAT 的最小值（微秒）
    base_e2e_latency = 1.0  # 固定基准值（毫秒）
    base_hop_latency = 1  # 每跳时延的固定基准值（毫秒）
    base_link_latency = 0.1  # 链路时延固定基准值（毫秒）
    base_queue_occupancy = 0  # 队列深度的固定基准值

    for i in range(num_samples):  # 使用 i 作为原始序号
        # 随机选择模态类型
        modal_type = random.choice(list(MODAL_TYPES.keys()))
        factor = MODAL_TYPES[modal_type]

        # 随机生成动态变化值
        dynamic_hop_latency = random.uniform(-0.1, 0.3)  # 每跳时延的动态变化值（毫秒）
        dynamic_link_latency = random.uniform(-0.01, 0.03)  # 每段链路的时延（毫秒）
        dynamic_queue_occupancy = random.uniform(0, 4)  # 队列深度的动态变化值
        base_hop_jitter = random.uniform(0.01, 0.1)  # 每跳的时延抖动（毫秒）
        is_anomaly = 0  # 是否异常

        # 计算每跳时延、链路时延和队列深度（不乘以 factor）
        hop_latency = base_hop_latency + dynamic_hop_latency  # 每跳时延
        link_latency = base_link_latency + dynamic_link_latency  # 每段链路时延
        queue_occupancy = base_queue_occupancy + dynamic_queue_occupancy  # 队列深度
        hop_jitter = base_hop_jitter  # 每跳时延抖动
        e2e_jitter = (NUM_HOPS ** 0.5) * hop_jitter  # 端到端时延抖动

        # 初始化 e2e_latency 和 IAT
        e2e_latency = NUM_HOPS * hop_latency + (NUM_HOPS - 1) * link_latency  # 端到端时延
        iat = base_iat * random.uniform(0.97, 1.03)  # 数据包到达间隔（微秒）

        # 模拟微突发流量
        if not is_burst and random.random() < anomaly_prob:
            # 开始微突发流量
            burst_peak = random.uniform(0.8, 1)
            is_burst = True
            burst_start = i
            burst_duration = random.randint(15, 50)  # 微突发流量的持续时间在 15~50 之间波动
            burst_end = min(i + burst_duration, num_samples)

        if is_burst and i < burst_end:
            # 在微突发流量期间，时延和队列深度对数增长
            progress = (i - burst_start) / burst_duration  # 微突发流量的进度（0 到 1）
            if progress < burst_peak:  # 前 80% 的时间对数增长
                # 使用对数函数计算增长因子
                vary_factor = fast_approaching(progress, 3)
                hop_latency *= (1 + vary_factor * burst_magnification)  # 时延对数增长
                link_latency *= (1 + vary_factor * burst_magnification)
                queue_occupancy += random.randint(3,4)
                queue_occupancy *= (1 + vary_factor * burst_magnification)  # 队列深度对数增长
            else:
                # 后 20% 的时间在最值附近波动
                hop_latency *= (1 + burst_magnification * burst_peak) * random.uniform(0.95, 1.05)
                link_latency *= (1 + 1 * burst_peak) * random.uniform(0.95, 1.05)
                queue_occupancy += random.randint(3, 4)
                queue_occupancy *= (1 + burst_magnification * burst_peak) * random.uniform(0.95, 1.05)

            # 计算端到端时延（e2e_latency）
            e2e_latency = NUM_HOPS * hop_latency + (NUM_HOPS - 1) * link_latency

            # IAT在前20%快速下降，然后趋于稳定
            if progress < 0.1:
                vary_factor = fast_approaching(progress*10, 4)
                iat = iat - iat * vary_factor * 0.8
            else:
                iat = min_iat * random.uniform(0.9, 1.1)

            # 时延抖动平滑增加
            hop_jitter *= (1 + 1 * progress)
            e2e_jitter = (NUM_HOPS ** 0.5) * hop_jitter
            is_anomaly = 1

        if is_burst and i >= burst_end:
            # 结束微突发流量
            is_burst = False

        # 在最后统一乘以模态类型的比例因子
        hop_latency *= factor
        link_latency *= factor
        queue_occupancy *= (factor - 1) * 0.3 + 1
        hop_jitter *= factor
        e2e_jitter *= factor
        e2e_latency *= factor

        # 将生成的数据添加到列表中，包括原始序号
        data.append([
            i,  # 原始序号
            modal_type,
            hop_latency,  # 每跳的时延（毫秒）
            e2e_latency,  # 端到端时延（毫秒）
            queue_occupancy,  # 队列深度
            iat,  # 数据包到达间隔（微秒）
            hop_jitter * 1000,  # 每跳的时延抖动（微秒）
            e2e_jitter * 1000,  # 端到端时延抖动（微秒）
            is_anomaly  # 是否异常
        ])

    return data


# 定义保存数据到 CSV 文件的函数
def save_to_csv(data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow([
            "Packet ID",  # 原始序号
            "Modal Type",
            "Hop Latency (ms)",
            "e2e Delay (ms)",
            "Queue Occupancy",
            "IAT (us)",
            "Hop Jitter (us)",
            "e2e Jitter (us)",
            "Is Anomaly"
        ])
        # 写入数据
        writer.writerows(data)


# 主函数
if __name__ == "__main__":
    # 生成 10000 条数据，异常概率为 0.1%
    num_samples = 1000
    anomaly_prob = 0.005  # 降低微突发流量的触发概率
    data = generate_network_data(num_samples, anomaly_prob)

    # 保存完整数据集到 CSV 文件
    filename = "int_data_anomaly_bmv2_all.csv"
    save_to_csv(data, filename)
    print(f"Generated {num_samples} samples with {anomaly_prob * 100}% anomaly probability and saved to {filename}.")

    # 按模态类型将数据分组并保存到单独的 CSV 文件
    for modal_type in MODAL_TYPES.keys():
        modal_data = [row for row in data if row[1] == modal_type]  # 注意：模态类型现在是第 2 列
        modal_filename = f"int_data_anomaly_bmv2_{modal_type}.csv"
        save_to_csv(modal_data, modal_filename)
        print(f"Saved {len(modal_data)} samples of {modal_type} to {modal_filename}.")