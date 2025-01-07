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

#ifndef __HEADERS__
#define __HEADERS__

#include "defines.p4"

header ethernet_t {
    mac_addr_t  dst_addr;
    mac_addr_t  src_addr;
    bit<16>     ether_type;
}

header ipv6_t {
    bit<4>   version;
    bit<8>   traffic_class;
    bit<20>  flow_label;
    bit<16>  payload_len;
    bit<8>   next_hdr;
    bit<8>   hop_limit;
    bit<128> src_addr;
    bit<128> dst_addr;
}

header ipv4_t {
    bit<4>    version;
    bit<4>    ihl;
    bit<6>    dscp;
    bit<2>    ecn;
    bit<16>   total_len;
    bit<16>   identification;
    bit<3>    flags;
    bit<13>   frag_offset;
    bit<8>    ttl;
    bit<8>    protocol;
    bit<16>   hdr_checksum;
    bit<32>   src_addr;
    bit<32>   dst_addr;
}

header flexip_t {
    bit<4>    version;
    bit<2>    src_format;
    bit<2>    dst_format;
    bit<12>   src_length;
    bit<12>   dst_length;
    bit<384>  src_addr;
    bit<384>  dst_addr;
}

header tcp_t {
    bit<16>  src_port;
    bit<16>  dst_port;
    bit<32>  seq_no;
    bit<32>  ack_no;
    bit<4>   data_offset;
    bit<3>   res;
    bit<3>   ecn;
    bit<6>   ctrl;
    bit<16>  window;
    bit<16>  checksum;
    bit<16>  urgent_ptr;
}

header udp_t {
    bit<16> src_port;
    bit<16> dst_port;
    bit<16> len;
    bit<16> checksum;
}

header icmp_t {
    bit<8>   type;
    bit<8>   icmp_code;
    bit<16>  checksum;
    bit<16>  identifier;
    bit<16>  sequence_number;
    bit<64>  timestamp;
}

header icmpv6_t {
    bit<8>   type;
    bit<8>   code;
    bit<16>  checksum;
}

header ndp_t {
    bit<32>      flags;
    ipv6_addr_t  target_ipv6_addr;
    // NDP option.
    bit<8>       type;
    bit<8>       length;
    bit<48>      target_mac_addr;
}

header ndn_prefix_t {
    bit<8> code;
    bit<8> len_code;
    bit<16> length;
}

header ndn_tlv_prefix_t {
    bit<8> code;
    bit<8> length;
}

header name_component_t {
    bit<8> code;
    bit<1> end;
    bit<7> length;
    // varbit
    bit<32> value;
}

struct name_tlv_t {
    ndn_tlv_prefix_t ndn_tlv_prefix;
    // 可嵌套多个component
    name_component_t[MAX_COMPONENTS] components;
}

header content_type_tlv_t {
    bit<8> code;
    bit<8> length;
    bit<16> value;
}

header freshness_period_tlv_t {
    bit<8> code;
    bit<8> length;
    bit<16> value;
}

header final_block_id_tlv_t {
    bit<8> code;
    bit<8> length;
    bit<16> value;
}

struct metaInfo_tlv_t {
    ndn_tlv_prefix_t ndn_tlv_prefix;
    // ContentType TLV
    content_type_tlv_t content_type_tlv;
    // FreshnessPeriod TLV
    freshness_period_tlv_t freshness_period_tlv;
    // FinalBlockId TLV
    final_block_id_tlv_t final_block_id_tlv;
}

header content_tlv_t {
    bit<8> code;
    bit<8> length;
    // varbit
    bit<16> value;
}

// ndn模态报文首部
struct ndn_t {
    ndn_prefix_t ndn_prefix;
    name_tlv_t name_tlv;
    metaInfo_tlv_t metaInfo_tlv;
    content_tlv_t content_tlv;
}

// 地理模态报文首部
header geo_t{
    bit<4> version;
    bit<4> nh_basic;
    bit<8> reserved_basic;
    bit<8> lt;
    bit<8> rhl;
    bit<4> nh_common;
    bit<4> reserved_common_a;
    bit<4> ht;  // 决定后续包型
    bit<4> hst;
    bit<8> tc;
    bit<8> flag;
    bit<16> pl;
    bit<8> mhl;
    bit<8> reserved_common_b;
}

header gbc_t{
    bit<16> sn;
    bit<16> reserved_gbc_a;
    bit<64> gnaddr;
    bit<32> tst;
    bit<32> lat;
    bit<32> longg;
    bit<1> pai;
    bit<15> s;
    bit<16> h;
    bit<32> geo_area_pos_lat; //lat 请求区域中心点的纬度
    bit<32> geo_area_pos_lon; //log 请求区域中心点的经度
    bit<16> disa;
    bit<16> disb;
    bit<16> angle;
    bit<16> reserved_gbc_b;
}


header beacon_t{
    bit<64> gnaddr;
    bit<32> tst;
    bit<32> lat;
    bit<32> longg;
    bit<1> pai;
    bit<15> s;
    bit<16> h;

}

// mf模态报文首部
header mf_t{
    bit<32> mf_type;
    bit<32> src_guid;
    bit<32> dst_guid;
}

// 身份模态报文首部
header id_t {
    bit<32> src_identity;
    bit<32> dst_identity;
}


// ---------------- int ----------------
// int填充垫片
header int_shim_t {
    bit<8> int_type;
    bit<8> rsvd1;
    bit<8> len;         // int首部的长度（四字节为单位）
    bit<6> dscp;        // ipv4的dscp字段
    bit<2> rsvd3;
}

// int首部
header int_header_t {
    bit<4> ver;
    bit<2> rep;
    bit<1> c;
    bit<1> e;
    bit<1> m;
    bit<7>  rsvd1;
    bit<3>  rsvd2;
    bit<5>  hop_metadata_len;   // 单个INT节点添加的元数据长度（4字节为单位）
    bit<8>  remaining_hop_cnt;  // 还可以添加int元数据的交换机个数
    bit<16> instruction_mask;   // instruction的bitmap
    bit<16> seq;  // rsvd3 - custom implementation of a sequence number
}

header int_switch_id_t {
    bit<32> switch_id;
}

header int_level1_port_ids_t {
    bit<32> ingress_port_id;
    bit<32> egress_port_id;
}

header int_hop_latency_t {
    bit<32> hop_latency;
}

header int_q_occupancy_t {
    bit<8> q_id;
    bit<24> q_occupancy;
}

header int_ingress_tstamp_t {
    bit<32> ingress_tstamp;
}

header int_egress_tstamp_t {
    bit<32> egress_tstamp;
}

header int_level2_port_ids_t {
    bit<32> ingress_port_id;
    bit<32> egress_port_id;
}

header int_egress_port_tx_util_t {
    bit<32> egress_port_tx_util;
}

header int_report_fixed_header_t {
    bit<4> ver;
    bit<4> len;
    bit<3> nprot;
    bit<5> rep_md_bits_high; // Split rep_md_bits to align to word boundaries
    bit<1> rep_md_bits_low;
    bit<6> reserved;
    bit<1> d;
    bit<1> q;
    bit<1> f;
    bit<6> hw_id;
    bit<32> switch_id;
    bit<32> seq_num;
    bit<32> ingress_tstamp;
}

struct int_metadata_t {
    bit<1>  source;    // is INT source functionality enabled
    bit<1>  sink;        // is INT sink functionality enabled
    bit<32> switch_id;  // INT switch id is configured by network controller
    bit<16> insert_byte_cnt;  // counter of inserted INT bytes
    bit<8>  int_hdr_word_len;  // counter of inserted INT words
    bit<1>  remove_int;           // indicator that all INT headers and data must be removed at egress for the processed packet 
    bit<16> sink_reporting_port;    // on which port INT reports must be send to INT collector
    bit<64> ingress_tstamp;   // pass ingress timestamp from Ingress pipeline to Egress pipeline
    bit<16> ingress_port;  // pass ingress port from Ingress pipeline to Egress pipeline 
}

header int_data_t {
    // Enough room for previous 4 nodes worth of data
    varbit<1600> data;
}

@controller_header("packet_in")
header packet_in_t {
    port_num_t ingress_port;
    bit<7> pad0;
}

@controller_header("packet_out")
header packet_out_t {
    port_num_t egress_port;
    bit<7> pad0;
}

struct headers_t {
    packet_out_t  packet_out;
    packet_in_t   packet_in;

    // INT report headers
    ethernet_t                report_ethernet;
    ipv4_t                    report_ipv4;
    udp_t                     report_udp;
    int_report_fixed_header_t report_fixed_header;

    ethernet_t    ethernet;
    ipv6_t        ipv6;
    ipv4_t        ipv4;
    flexip_t      flexip;
    id_t          id;
    mf_t          mf;
    geo_t         geo;
    gbc_t         gbc;
    beacon_t      beacon;
    ndn_t         ndn;
    tcp_t         tcp;
    udp_t         udp;
    icmpv6_t      icmpv6;
    ndp_t         ndp;

    // INT headers
    int_shim_t              int_shim;
    int_header_t              int_header;
  
    // local INT node metadata
    int_egress_port_tx_util_t int_egress_port_tx_util;
    int_egress_tstamp_t       int_egress_tstamp;
    int_hop_latency_t         int_hop_latency;
    int_ingress_tstamp_t      int_ingress_tstamp;
    int_level1_port_ids_t     int_level1_port_ids;
    int_level2_port_ids_t     int_level2_port_ids;
    int_q_occupancy_t         int_q_occupancy;
    int_switch_id_t           int_switch_id;

    // INT metadata of previous nodes
    int_data_t                int_data;
}

struct local_metadata_t {
    int_metadata_t  int_metadata;
    l4_port_t       l4_src_port;
    l4_port_t       l4_dst_port;
    bit<6>          l4_dscp;
    bit<8>          name_tlv_length;
    bool            is_multicast;
    next_hop_id_t   next_hop_id;
    bit<16>         selector;
    bool            compute_checksum;
}

#endif
