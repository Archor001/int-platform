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

data = "ABCDEF"  # 数据包负载

# 发包速率（单位：包/秒）
packet_rate = 1500  # 默认 1000 包/秒

# 获取网络接口
interface = [i for i in get_if_list() if "eth0" in i][0]
s = conf.L2socket(iface=interface)

# 构造IPv4数据包
p1 = Ether(dst=dst_mac, src=src_mac) / IP(frag=0, dst=dst_ip, src=src_ip)
p1 = p1 / UDP(sport=sport, dport=dport) / Raw(load=data)

# 构造ID数据包
p2 = Ether(dst=dst_mac, src=src_mac, type=ETHERTYPE_ID) / Raw(load=struct.pack("!BLL", 0x01, src_identity, dst_identity))
p2 = p2 / UDP(sport=sport, dport=dport) / Raw(load=data)

# 构造MF数据包
p3 = Ether(dst=dst_mac, src=src_mac, type=ETHERTYPE_MF) / Raw(load=struct.pack("!BLLL", 0x01, 0x0000001, src_mfguid, dst_mfguid))
p3 = p3 / UDP(sport=sport, dport=dport) / Raw(load=data)

# 构造NDN数据包
p4 = Ether(dst=dst_mac, src=src_mac, type=ETHERTYPE_NDN) / Raw(load=struct.pack("!BLLLLLLLLL", 0x01, 0x6fd0020, 0x80c0804, src_ndn_name,
                                     0x08840000 | ((dst_ndn_name >> 16) & 0xffff), (((dst_ndn_name & 0xffff)) << 16) | 0x1e00, 
                                     0x18020000, 0x19020000,0x1b020000,0x1a020000 | ndn_content))
p4 = p4 / UDP(sport=sport, dport=dport) / Raw(load=data)

packets = []  # 用于存储所有数据包

if __name__ == "__main__":
    pkt_cnt = 0
    last_sec = time.time()
    interval = 1.0 / packet_rate  # 计算每个数据包的时间间隔

    try:
        while True:
            start_time = time.time()  # 记录发送开始时间

            s.send(p1)
            packets.append(p1)  # 将数据包添加到列表中
            pkt_cnt += 1

            # 计算实际发送时间，并调整等待时间
            elapsed_time = time.time() - start_time
            sleep_time = max(0, interval - elapsed_time)  # 确保 sleep_time 不为负
            time.sleep(sleep_time)

            # 每秒打印一次发包速率
            if time.time() - last_sec > 1.0:
                print("Pkt/s:", pkt_cnt)
                pkt_cnt = 0
                last_sec = time.time()

    except KeyboardInterrupt:
        print("Writing packets to pcap file...")
        wrpcap('send_udp_flow.pcap', packets)  # 将所有数据包写入 pcap 文件
        print("Done.")