#!/usr/bin/python
from scapy.all import sniff, wrpcap, get_if_list
import sys
import time

# 定义网卡名称
interface = [i for i in get_if_list() if "veth_dp_3" in i][0]

# 定义一个列表来存储捕获的数据包
packets = []

# 定义一个回调函数，用于处理捕获到的每个数据包
def packet_callback(packet):
    global packets
    packets.append(packet)

# 定义一个函数来捕获数据包
def capture_packets():
    print("Interface is:", interface)
    print("Starting packet capture on veth_dp_3...")
    # 使用 sniff 函数捕获数据包，每次捕获持续1秒钟
    sniff(iface=interface, prn=packet_callback, store=False, timeout=1)
    # 打印捕获到的数据包数量
    print("Received {} packets in the last second.".format(len(packets)))
    # 将捕获的数据包写入到 pcap 文件中
    wrpcap('recv_udp_flow.pcap', packets)
    print("Packets saved to recv_udp_flow.pcap")

if __name__ == "__main__":
    capture_packets()