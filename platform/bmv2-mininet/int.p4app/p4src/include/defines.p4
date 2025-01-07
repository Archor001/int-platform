/*
 * Copyright 2017-present Open Networking Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define __DEFINES__
#define MAX_PORTS 511
#define CPU_PORT 255
#define MAX_COMPONENTS 8
#define CPU_CLONE_SESSION_ID 99

#define PKT_INSTANCE_TYPE_NORMAL 0
#define PKT_INSTANCE_TYPE_INGRESS_CLONE 1
#define PKT_INSTANCE_TYPE_EGRESS_CLONE 2
#define PKT_INSTANCE_TYPE_COALESCED 3
#define PKT_INSTANCE_TYPE_INGRESS_RECIRC 4
#define PKT_INSTANCE_TYPE_REPLICATION 5
#define PKT_INSTANCE_TYPE_RESUBMIT 6

#define _BOOL bool
#define _TRUE true
#define _FALSE false

typedef bit<9>   port_num_t;
typedef bit<48>  mac_addr_t;
typedef bit<16>  mcast_group_id_t;
typedef bit<32>  ipv4_addr_t;
typedef bit<128> ipv6_addr_t;
typedef bit<16>  l4_port_t;
typedef bit<16>  next_hop_id_t;

const bit<16> ETHERTYPE_IPV6 = 0x86dd;
const bit<16> ETHERTYPE_IPV4 = 0x0800;
const bit<16> ETHERTYPE_ID = 0x0812;
const bit<16> ETHERTYPE_GEO = 0x8947;
const bit<16> ETHERTYPE_MF = 0x27c0;
const bit<16> ETHERTYPE_NDN = 0x8624;
const bit<16> ETHERTYPE_FLEXIP = 0x3690;

const bit<4> TYPE_geo_beacon = 1;
const bit<4> TYPE_geo_gbc = 4;

const bit<8> IP_PROTO_TCP = 6;
const bit<8> IP_PROTO_UDP = 17;
const bit<8> IP_PROTO_ICMPV6 = 58;

const mac_addr_t IPV6_MCAST_01 = 0x33_33_00_00_00_01;

const bit<8> ICMP6_TYPE_NS = 135;
const bit<8> ICMP6_TYPE_NA = 136;
const bit<8> NDP_OPT_TARGET_LL_ADDR = 2;
const bit<32> NDP_FLAG_ROUTER = 0x80000000;
const bit<32> NDP_FLAG_SOLICITED = 0x40000000;
const bit<32> NDP_FLAG_OVERRIDE = 0x20000000;

typedef bit<8> MeterColor;
const MeterColor MeterColor_GREEN = 8w0;
const MeterColor MeterColor_YELLOW = 8w1;
const MeterColor MeterColor_RED = 8w2;

// int
const bit<6>  IPv4_DSCP_INT = 0x20;  // indicates an INT header in the packet
const bit<8>  INT_TYPE_HOP_BY_HOP = 1;   // HOP_BY_HOP INT type 
const bit<16> INT_SHIM_HEADER_LEN_BYTES = 4;
const bit<16> INT_HEADER_LEN_BYTES = 8;
const bit<16> INT_ALL_HEADER_LEN_BYTES = INT_SHIM_HEADER_LEN_BYTES + INT_HEADER_LEN_BYTES;
const bit<4>  INT_VERSION = 1;
const bit<4> INT_REPORT_HEADER_LEN_WORDS = 4;
const bit<4> INT_REPORT_VERSION = 1;