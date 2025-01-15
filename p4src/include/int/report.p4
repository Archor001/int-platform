/*
 * Copyright 2020-2021 PSNC, FBK
 *
 * Author: Bartosz Krakowiak, Damu Ding
 *
 * Created in the GN4-3 project.
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

// Code adapted from:
// - https://github.com/baru64/int-p4/blob/master/int.p4app/p4src/int_report.p4

// register to store seq_num
register<bit<32>> (1) report_seq_num_register;

control Int_report(inout headers_t hdr, inout local_metadata_t meta, inout standard_metadata_t standard_metadata) {
        bit<32> seq_num_value = 0;

        // INT Report structure
        // [Eth][IP][UDP][INT RAPORT HDR][ETH][IP][UDP/TCP][INT SHIM][INT DATA]

        action send_report(bit<48> dp_mac, bit<32> dp_ip, bit<48> collector_mac, bit<32> collector_ip, bit<16> collector_port) {

            // Ethernet **********************************************************
            hdr.report_ethernet.setValid();
            hdr.report_ethernet.dst_addr = collector_mac;
            hdr.report_ethernet.src_addr = dp_mac;
            hdr.report_ethernet.ether_type = 0x0800;

            // IPv4 **************************************************************
            hdr.report_ipv4.setValid();
            hdr.report_ipv4.version = 4;
            hdr.report_ipv4.ihl = 5;
            hdr.report_ipv4.dscp = 0;
            hdr.report_ipv4.ecn = 0;

            // ipv4 header (20) + udp header (8) + report header (16) + eth header (14) + modal header + int data len
            hdr.report_ipv4.total_len = (bit<16>)(20 + 8 + 14)
                + ((bit<16>)(INT_REPORT_HEADER_LEN_WORDS)<<2)
                + (((bit<16>)hdr.int_shim.len) << 2);
            
            // modal header len
            if (hdr.ipv4.isValid()) {
                hdr.report_ipv4.total_len = hdr.report_ipv4.total_len + 20;
            }
            if (hdr.id.isValid()) {
                hdr.report_ipv4.total_len = hdr.report_ipv4.total_len + 9;
            }
            if (hdr.mf.isValid()) {
                hdr.report_ipv4.total_len = hdr.report_ipv4.total_len + 13;
            }
            if (hdr.geo.isValid()) {
                hdr.report_ipv4.total_len = hdr.report_ipv4.total_len + 57;
            }
            if (hdr.ndn.ndn_prefix.isValid()) {
                hdr.report_ipv4.total_len = hdr.report_ipv4.total_len + 37;
            }
            // if (hdr.flexip.isValid()) {
            //     hdr.report_ipv4.total_len = hdr.report_ipv4.total_len + 20
            // }
                
            // add size of original tcp/udp header
            if (hdr.tcp.isValid()) {
                hdr.report_ipv4.total_len = hdr.report_ipv4.total_len
                    + (((bit<16>)hdr.tcp.data_offset) << 2);

            } else {
                hdr.report_ipv4.total_len = hdr.report_ipv4.total_len + 8;
            }
            hdr.report_ipv4.identification = 0;
            hdr.report_ipv4.flags = 0;
            hdr.report_ipv4.frag_offset = 0;
            hdr.report_ipv4.ttl = 64;
            hdr.report_ipv4.protocol = 17; // UDP
            hdr.report_ipv4.src_addr = dp_ip;
            hdr.report_ipv4.dst_addr = collector_ip;

            // UDP ***************************************************************
            hdr.report_udp.setValid();
            hdr.report_udp.src_port = 0;
            hdr.report_udp.dst_port = collector_port;
            hdr.report_udp.len = hdr.report_ipv4.total_len - 20;

            // INT report fixed header ************************************************/
            // INT report version 1.0
            hdr.report_fixed_header.setValid();
            hdr.report_fixed_header.ver = INT_REPORT_VERSION;
            hdr.report_fixed_header.len = INT_REPORT_HEADER_LEN_WORDS;

            hdr.report_fixed_header.nprot = 0; // 0 for Ethernet
            hdr.report_fixed_header.rep_md_bits_high = 0;
            hdr.report_fixed_header.rep_md_bits_low = 0;
            if (meta.modal_type == ETHERTYPE_IPV4) {
                hdr.report_fixed_header.reserved = 0;
            }
            if (meta.modal_type == ETHERTYPE_ID) {
                hdr.report_fixed_header.reserved = 1;
            }
            if (meta.modal_type == ETHERTYPE_GEO) {
                hdr.report_fixed_header.reserved = 2;
            }
            if (meta.modal_type == ETHERTYPE_MF) {
                hdr.report_fixed_header.reserved = 3;
            }
            if (meta.modal_type == ETHERTYPE_NDN) {
                hdr.report_fixed_header.reserved = 4;
            }
            if (meta.modal_type == ETHERTYPE_FLEXIP) {
                hdr.report_fixed_header.reserved = 5;
            }
            hdr.report_fixed_header.d = 0;
            hdr.report_fixed_header.q = 0;
            // f - indicates that report is for tracked flow, INT data is present
            hdr.report_fixed_header.f = 1;
            // hw_id - specific to the switch, e.g. id of linecard
            hdr.report_fixed_header.hw_id = 0;
            hdr.report_fixed_header.switch_id = meta.int_metadata.switch_id;
            report_seq_num_register.read(seq_num_value, 0);
            hdr.report_fixed_header.seq_num = seq_num_value;
            report_seq_num_register.write(0, seq_num_value + 1);

            hdr.report_fixed_header.ingress_tstamp = (bit<32>)standard_metadata.ingress_global_timestamp;

            // Original packet headers, INT shim and INT data come after report header.
            // drop all data besides int report and report eth header
            truncate((bit<32>)hdr.report_ipv4.total_len + 14);
            }
            table tb_int_reporting {
                actions = {
                    send_report;
                }
                size = 512;
            }

        apply {
            tb_int_reporting.apply();
        }
    }
