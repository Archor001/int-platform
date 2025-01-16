#!/usr/bin/python

# Copyright 2020-2021 PSNC
# Author: Damian Parniewicz
#
# Created in the GN4-3 project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from scapy.all import Ether, IP, sendp, get_if_hwaddr, get_if_list, TCP, Raw, UDP, wrpcap
from scapy.config import conf
import sys
import time
import struct
import random

# 配置参数
src_mac = "00:00:00:00:01:01"
dst_mac = "00:00:00:00:04:04"

src_ip = "10.0.1.1"
dst_ip = "10.0.4.4"

src_identity = 202271720
dst_identity = 202271723

src_mfguid = 1
dst_mfguid = 4

ETHERTYPE_ID = 0x0812
ETHERTYPE_MF = 0x27c0
ETHERTYPE_NDN = 0x8624

src_ndn_name = 13849245
dst_ndn_name = 13849248
ndn_content = 2048

sport = 0x11FF
dport = 0x22FF

data1 = "AAAAAAAAAAAAAAAAAAAHHHHHHHHHHHHHHHH"  # IPv4数据包负载
data2 = "GGGGGGYYYYYYY"  # ID数据包负载
data3 = "KKKKKKKKKOOOOOOOOO"  # MF数据包负载
data4 = "NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM"  # NDN数据包负载

# 获取网络接口
interface = [i for i in get_if_list() if "eth0" in i][0]
s = conf.L2socket(iface=interface)

# 构造IPv4数据包
p1 = Ether(dst=dst_mac, src=src_mac) / IP(frag=0, dst=dst_ip, src=src_ip)
p1 = p1 / UDP(sport=sport, dport=dport) / Raw(load=data1)

# 构造ID数据包
p2 = Ether(dst=dst_mac, src=src_mac, type=ETHERTYPE_ID) / Raw(load=struct.pack("!BLL", 0x01, src_identity, dst_identity))
p2 = p2 / UDP(sport=sport, dport=dport) / Raw(load=data2)

# 构造MF数据包
p3 = Ether(dst=dst_mac, src=src_mac, type=ETHERTYPE_MF) / Raw(load=struct.pack("!BLLL", 0x01, 0x0000001, src_mfguid, dst_mfguid))
p3 = p3 / UDP(sport=sport, dport=dport) / Raw(load=data3)

# 构造NDN数据包
p4 = Ether(dst=dst_mac, src=src_mac, type=ETHERTYPE_NDN) / Raw(load=struct.pack("!BLLLLLLLLL", 0x01, 0x6fd0020, 0x80c0804, src_ndn_name,
                                     0x08840000 | ((dst_ndn_name >> 16) & 0xffff), (((dst_ndn_name & 0xffff)) << 16) | 0x1e00, 
                                     0x18020000, 0x19020000,0x1b020000,0x1a020000 | ndn_content))
p4 = p4 / UDP(sport=sport, dport=dport) / Raw(load=data4)

# 定义数据包列表
packets = []  # 用于存储所有数据包
max_packets = 20000  # 最大数据包数量

if __name__ == "__main__":
    pkt_cnt = 0
    last_sec = time.time()

    try:
        while len(packets) < max_packets:
            # 随机选择一个数据包类型
            pkt = random.choice([p1, p2, p3, p4])
            packets.append(pkt)  # 将数据包添加到列表中
            pkt_cnt += 1

            # 每秒打印一次发包速率
            if time.time() - last_sec > 1.0:
                # print(f"Packets: {len(packets)}/{max_packets}, Pkt/s: {pkt_cnt}")
                pkt_cnt = 0
                last_sec = time.time()

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        # 将数据包写入 pcap 文件
        print("Writing {} packets to pcap file...".format(len(packets)))
        wrpcap('send_udp_flow.pcap', packets)  # 将所有数据包写入 pcap 文件
        print("Done.")