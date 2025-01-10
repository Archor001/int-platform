/*
 * Copyright 2020-2021 PSNC, FBK
 *
 * Author: Damian Parniewicz, Damu Ding
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

#include "headers.p4"

parser ParserImpl(packet_in packet, out headers_t hdr, inout local_metadata_t meta, inout standard_metadata_t standard_metadata) {
    state start {
       transition parse_ethernet;
    }
    state parse_ethernet {
        packet.extract(hdr.ethernet);
        meta.modal_type = (bit<32>)hdr.ethernet.ether_type;
        transition select(hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4: parse_ipv4;
            ETHERTYPE_IPV6: parse_ipv6;
            ETHERTYPE_ID: parse_id;
            ETHERTYPE_GEO: parse_geo;
            ETHERTYPE_MF: parse_mf;
            ETHERTYPE_NDN: parse_ndn;
            ETHERTYPE_FLEXIP: parse_flexip;
            default: accept;
        }
    }
    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        meta.l4_dscp = hdr.ipv4.dscp;
        transition select(hdr.ipv4.protocol) {
            IP_PROTO_TCP: parse_tcp;
            IP_PROTO_UDP: parse_udp;
            default: accept;
        }
    }

    state parse_ipv6 {
        packet.extract(hdr.ipv6);
        transition select(hdr.ipv6.next_hdr) {
            IP_PROTO_TCP:    parse_tcp;
            IP_PROTO_UDP:    parse_udp;
            default: accept;
        }
    }

    state parse_flexip {
        packet.extract(hdr.flexip);
        transition accept;
    }

    state parse_id {
        packet.extract(hdr.id);
        transition accept;
    }

    state parse_mf {
        packet.extract(hdr.mf);
        transition accept;
    }

    state parse_geo {
        packet.extract(hdr.geo);
        transition select(hdr.geo.ht) { //
            TYPE_geo_beacon: parse_beacon; //0x01
            TYPE_geo_gbc: parse_gbc; //0x04
            default: accept;
        }
    }

    state parse_beacon{
        packet.extract(hdr.beacon);
        transition accept;
    }

    state parse_gbc{
        packet.extract(hdr.gbc);
        transition accept;
    }

    state parse_ndn {
        packet.extract(hdr.ndn.ndn_prefix);
        transition parse_ndn_name;
    }

    state parse_ndn_name {
        packet.extract(hdr.ndn.name_tlv.ndn_tlv_prefix);
        meta.name_tlv_length = hdr.ndn.name_tlv.ndn_tlv_prefix.length;
        transition parse_ndn_name_components;
    }

    state parse_ndn_name_components {
        packet.extract(hdr.ndn.name_tlv.components.next);
        transition select(hdr.ndn.name_tlv.components.last.end) {
            0: parse_ndn_name_components;
            1: parse_ndn_metainfo;
        }
    }

    state parse_ndn_metainfo {
        packet.extract(hdr.ndn.metaInfo_tlv.ndn_tlv_prefix);
        packet.extract(hdr.ndn.metaInfo_tlv.content_type_tlv);
        packet.extract(hdr.ndn.metaInfo_tlv.freshness_period_tlv);
        packet.extract(hdr.ndn.metaInfo_tlv.final_block_id_tlv);
        transition parse_ndn_content;
    }

    state parse_ndn_content {
        packet.extract(hdr.ndn.content_tlv);
        transition accept;
    }

    state parse_tcp {
        packet.extract(hdr.tcp);
        meta.l4_src_port = hdr.tcp.src_port;
        meta.l4_dst_port = hdr.tcp.dst_port;
        transition select(meta.l4_dscp) {
            IPv4_DSCP_INT: parse_int;
            default: accept;
        }
    }
    state parse_udp {
        packet.extract(hdr.udp);
        meta.l4_src_port = hdr.udp.src_port;
        meta.l4_dst_port = hdr.udp.dst_port;
        transition select(meta.l4_dscp, hdr.udp.dst_port){
            (6w0x20 &&& 6w0x3f, 16w0x0 &&& 16w0x0): parse_int;
            default: accept;
        }
    }
    state parse_int {
        packet.extract(hdr.int_shim);
        packet.extract(hdr.int_header);
        transition accept;
    }
}

control DeparserImpl(packet_out packet, in headers_t hdr) {
    apply {
        // raport headers
        packet.emit(hdr.report_ethernet);
        packet.emit(hdr.report_ipv4);
        packet.emit(hdr.report_udp);
        packet.emit(hdr.report_fixed_header);
        
        // original headers
        packet.emit(hdr.ethernet);
        packet.emit(hdr.ndn);
	    packet.emit(hdr.mf);
        packet.emit(hdr.id);
        packet.emit(hdr.flexip);
        packet.emit(hdr.geo);
	    packet.emit(hdr.gbc);
	    packet.emit(hdr.beacon);
        packet.emit(hdr.ipv6);
        packet.emit(hdr.ipv4);
        packet.emit(hdr.tcp);
        packet.emit(hdr.udp);
        packet.emit(hdr.icmpv6);
        packet.emit(hdr.ndp);
        
        // INT headers
        packet.emit(hdr.int_shim);
        packet.emit(hdr.int_header);
        
        // local INT node metadata
        packet.emit(hdr.int_switch_id);     // 4 bytes
        packet.emit(hdr.int_modal_type);    // 4 bytes
        packet.emit(hdr.int_port_ids);      // 8 bytes
        packet.emit(hdr.int_hop_latency);   // 4 bytes
        packet.emit(hdr.int_q_occupancy);   // 4 bytes
        packet.emit(hdr.int_ingress_tstamp);  // 4 bytes
        packet.emit(hdr.int_egress_tstamp);   // 4 bytes
    }
}

control verifyChecksum(inout headers_t hdr, inout local_metadata_t meta) {
    apply {
    }
}

control computeChecksum(inout headers_t hdr, inout local_metadata_t meta) {
    apply {
        update_checksum(
            hdr.ipv4.isValid(),
            {
                hdr.ipv4.version,
                hdr.ipv4.ihl,
                hdr.ipv4.dscp,
                hdr.ipv4.ecn,
                hdr.ipv4.total_len,
                hdr.ipv4.identification,
                hdr.ipv4.flags,
                hdr.ipv4.frag_offset,
                hdr.ipv4.ttl,
                hdr.ipv4.protocol,
                hdr.ipv4.src_addr,
                hdr.ipv4.dst_addr
            },
            hdr.ipv4.hdr_checksum,
            HashAlgorithm.csum16
        );
        
        update_checksum(
            hdr.report_ipv4.isValid(),
            {
                hdr.report_ipv4.version,
                hdr.report_ipv4.ihl,
                hdr.report_ipv4.dscp,
                hdr.report_ipv4.ecn,
                hdr.report_ipv4.total_len,
                hdr.report_ipv4.identification,
                hdr.report_ipv4.flags,
                hdr.report_ipv4.frag_offset,
                hdr.report_ipv4.ttl,
                hdr.report_ipv4.protocol,
                hdr.report_ipv4.src_addr,
                hdr.report_ipv4.dst_addr
            },
            hdr.report_ipv4.hdr_checksum,
            HashAlgorithm.csum16
        );
        
        update_checksum_with_payload(
            hdr.udp.isValid(), 
            {  hdr.ipv4.src_addr, 
                hdr.ipv4.dst_addr, 
                8w0, 
                hdr.ipv4.protocol, 
                hdr.udp.len, 
                hdr.udp.src_port, 
                hdr.udp.dst_port, 
                hdr.udp.len 
            }, 
            hdr.udp.checksum, 
            HashAlgorithm.csum16
        ); 

        update_checksum_with_payload(
            hdr.udp.isValid() && hdr.int_header.isValid() , 
            {  hdr.ipv4.src_addr, 
                hdr.ipv4.dst_addr, 
                8w0, 
                hdr.ipv4.protocol, 
                hdr.udp.len, 
                hdr.udp.src_port, 
                hdr.udp.dst_port, 
                hdr.udp.len,
                hdr.int_shim,
                hdr.int_header,
                hdr.int_switch_id,
                hdr.int_modal_type,
                hdr.int_port_ids,
                hdr.int_hop_latency,
                hdr.int_q_occupancy,
                hdr.int_ingress_tstamp,
                hdr.int_egress_tstamp
            },
            hdr.udp.checksum, 
            HashAlgorithm.csum16
        );
    }
}