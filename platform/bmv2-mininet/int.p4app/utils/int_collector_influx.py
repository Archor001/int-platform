import argparse
import time
import socket
import struct
import binascii
import pprint
import logging
from copy import copy
import io
from influxdb import InfluxDBClient

log_format = "[%(asctime)s] [%(levelname)s] - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format, filename="/tmp/bmv2-mininet/int.p4app/p4app_logs/int_collector.log")
logger = logging.getLogger('int_collector')


def parse_params():
    parser = argparse.ArgumentParser(description='InfluxBD INTCollector client.')

    parser.add_argument("-i", "--int_port", default=54321, type=int,
        help="Destination port of INT Telemetry reports")

    parser.add_argument("-H", "--host", default="localhost",
        help="InfluxDB server address")

    parser.add_argument("-D", "--database", default="int_telemetry_db",
        help="Database name")

    parser.add_argument("-p", "--period", default=1, type=int,
        help="Time period to push data in normal condition")

    parser.add_argument("-d", "--debug_mode", default=0, type=int,
        help="Set to 1 to print debug information")

    return parser.parse_args()



class HopMetadata:
    def __init__(self, data, ins_map, int_version=1):
        self.data = data
        self.ins_map = ins_map
        
        self.__parse_switch_id()
        self.__parse_modal_type()
        self.__parse_ports()
        self.__parse_hop_latency()
        self.__parse_queue_occupancy()
        self.__parse_ingress_timestamp()
        self.__parse_egress_timestamp()
        
    def __parse_switch_id(self):
        if self.ins_map & 0x80:
            self.switch_id = int.from_bytes(self.data.read(4), byteorder='big')
            logger.debug('parse switch id: %d' % self.switch_id)
        
    def __parse_modal_type(self):
        if self.ins_map & 0x40:
            self.modal_type = int.from_bytes(self.data.read(4), byteorder='big')
            logger.debug('parse modal type: %d' % self.modal_type)

    def __parse_ports(self):
        if self.ins_map & 0x20:
            self.l1_ingress_port_id = int.from_bytes(self.data.read(4), byteorder='big')
            self.l1_egress_port_id = int.from_bytes(self.data.read(4), byteorder='big')
            logger.debug('parse ingress port: %d, egress_port: %d' % (self.l1_ingress_port_id , self.l1_egress_port_id))
        
    def __parse_hop_latency(self):
        if self.ins_map & 0x10:
            self.hop_latency  = int.from_bytes(self.data.read(4), byteorder='big')
            logger.debug('parse hop latency: %d' %  self.hop_latency)
    
    def __parse_queue_occupancy(self):
        if self.ins_map & 0x04:
            self.queue_occupancy_id = int.from_bytes(self.data.read(1), byteorder='big')
            self.queue_occupancy = int.from_bytes(self.data.read(3), byteorder='big')
            logger.debug('parse queue_occupancy_id: %d, queue_occupancy: %d' % (self.queue_occupancy_id, self.queue_occupancy))
            
    def __parse_ingress_timestamp(self):
        if self.ins_map & 0x02:
            self.ingress_timestamp  = int.from_bytes(self.data.read(4), byteorder='big')
            logger.debug('parse ingress_timestamp: %d' %  self.ingress_timestamp)
            
    def __parse_egress_timestamp(self):
        if self.ins_map & 0x01:
            self.egress_timestamp  = int.from_bytes(self.data.read(4), byteorder='big')
            logger.debug('parse egress_timestamp: %d' %  self.egress_timestamp)
            
    def unread_data(self):
        return self.data
            
    def __str__(self):
        attrs = vars(self)
        try:
            del attrs['data']
            del attrs['ins_map']
        except Exception as e: 
            logger.error(e)
        return pprint.pformat(attrs)



def ip2str(ip):
    return "{}.{}.{}.{}".format(ip[0],ip[1],ip[2],ip[3])

def modal_protocol(value):
    if value == 0:
        return 6
    else:
        return 17

def modal2str(hdr, modal, which):
    if modal == 0:      # ipv4
        if which == 0:
            return ip2str(hdr[12:16])
        if which == 1:
            return ip2str(hdr[16:20])
    if modal == 1:      # id
        if which == 0:
            return "0x" + hdr[1:5].hex()
        if which == 1:
            return "0x" + hdr[5:9].hex()
    # if modal == 2:      # geo
    if modal == 3:      # mf
        if which == 0:
            return "0x" + hdr[5:9].hex()
        if which == 1:
            return "0x" + hdr[9:13].hex()
    if modal == 4:      # ndn
        if which == 0:
            return "0x" + hdr[9:13].hex()
        if which == 1:
            return "0x" + hdr[15:19].hex()
    # if modal == 5:      # flexip


class IntReport():
    def __init__(self, data):
        orig_data = data
        #data = struct.unpack("!%dB" % len(data), data)
        '''
        header int_report_fixed_header_t {
            bit<4> ver;
            bit<4> len;
            bit<3> nprot;
            bit<6> rep_md_bits;
            bit<6> reserved;
            bit<1> d;
            bit<1> q;
            bit<1> f;
            bit<6> hw_id;
            bit<32> switch_id;
            bit<32> seq_num;
            bit<32> ingress_tstamp;
        }
        const bit<8> REPORT_FIXED_HEADER_LEN = 16;
        '''
        
        # report header
        self.int_report_hdr = data[:16]
        self.ver = self.int_report_hdr[0] >> 4
        
        if self.ver != 1:
            logger.error("Unsupported INT report version %s - skipping report" % self.int_version)
            raise Exception("Unsupported INT report version %s - skipping report" % self.int_version)
        
        self.len = self.int_report_hdr[0] & 0x0f
        self.nprot = self.int_report_hdr[1] >> 5
        self.rep_md_bits = (self.int_report_hdr[1] & 0x1f) + (self.int_report_hdr[2] >> 7)
        self.reserved = (self.int_report_hdr[2] & 0x7e) >> 1
        self.d = self.int_report_hdr[2] & 0x01
        self.q = self.int_report_hdr[3] >> 7
        self.f = (self.int_report_hdr[3] >> 6) & 0x01
        self.hw_id = self.int_report_hdr[3] & 0x3f
        self.switch_id, self.seq_num, self.ingress_tstamp = struct.unpack('!3I', orig_data[4:16])

        # flow id
        self.flow_id = {}
        protocol = 0
        modal_length = 0
        if self.reserved == 0:  # ipv4
            self.modal_hdr = data[30:50]
            self.udp_hdr = data[50:58]
            protocol = self.modal_hdr[9]
            modal_length = 20
            self.flow_id = {
                'src': modal2str(self.modal_hdr, self.reserved, 0),
                'dst': modal2str(self.modal_hdr, self.reserved, 1), 
                'scrp': struct.unpack('!H', self.udp_hdr[:2])[0],
                'dstp': struct.unpack('!H', self.udp_hdr[2:4])[0],
                'protocol': protocol,       
            }
        if self.reserved == 1:  # id
            self.modal_hdr = data[30:39]
            self.udp_hdr = data[39:47]
            protocol = modal_protocol(self.modal_hdr[0])
            modal_length = 9
            self.flow_id = {
                'src': modal2str(self.modal_hdr, self.reserved, 0),
                'dst': modal2str(self.modal_hdr, self.reserved, 1), 
                'scrp': struct.unpack('!H', self.udp_hdr[:2])[0],
                'dstp': struct.unpack('!H', self.udp_hdr[2:4])[0],
                'protocol': protocol,       
            }
        # if self.reserved == 2:  # geo
        #     self.modal_hdr = data[30:39]
        #     self.udp_hdr = data[39:47]
        #     protocol = modal_protocol(self.modal_hdr[0])
        if self.reserved == 3:  # mf
            self.modal_hdr = data[30:43]
            self.udp_hdr = data[43:51]
            protocol = modal_protocol(self.modal_hdr[0])
            modal_length = 13
            self.flow_id = {
                'src': modal2str(self.modal_hdr, self.reserved, 0),
                'dst': modal2str(self.modal_hdr, self.reserved, 1), 
                'scrp': struct.unpack('!H', self.udp_hdr[:2])[0],
                'dstp': struct.unpack('!H', self.udp_hdr[2:4])[0],
                'protocol': protocol,       
            }
        if self.reserved == 4:  # ndn
            self.modal_hdr = data[30:67]
            self.udp_hdr = data[67:71]
            protocol = modal_protocol(self.modal_hdr[0])
            modal_length = 37
            self.flow_id = {
                'src': modal2str(self.modal_hdr, self.reserved, 0),
                'dst': modal2str(self.modal_hdr, self.reserved, 1), 
                'scrp': struct.unpack('!H', self.udp_hdr[:2])[0],
                'dstp': struct.unpack('!H', self.udp_hdr[2:4])[0],
                'protocol': protocol,       
            }
        # if self.reserved == 5:  # flexip
        #     self.modal_hdr = data[30:39]
        #     self.udp_hdr = data[39:47]
        #     protocol = modal_protocol(self.modal_hdr[0])

        # int_offset = report length(16) + eth length(14) + modal length + udp length(8)
        offset = 16 + 14 + modal_length + 8

        '''
        header intl4_shim_t {
            bit<8> int_type;
            bit<8> rsvd1;
            bit<8> len;   // the length of all INT headers in 4-byte words
            bit<6> rsvd2;  // dscp not put here
            bit<2> rsvd3;
        }
        const bit<16> INT_SHIM_HEADER_LEN_BYTES = 4;
        '''
        # int shim
        logger.info("reserved:%d, offset:%d, flow_id:%s" % (self.reserved, offset, self.flow_id))
        self.int_shim = data[offset:offset + 4]
        self.int_type = self.int_shim[0]
        self.int_data_len = int(self.int_shim[2]) - 3
        
        if self.int_type != 1: 
            logger.error("Unsupported INT type %s - skipping report" % self.int_type)
            raise Exception("Unsupported INT type %s - skipping report" % self.int_type)
  
        '''  INT header version 0.4     
        header int_header_t {
            bit<2> ver;
            bit<2> rep;
            bit<1> c;
            bit<1> e;
            bit<5> rsvd1;
            bit<5> ins_cnt;  // the number of instructions that are set in the instruction mask
            bit<8> max_hops; // maximum number of hops inserting INT metadata
            bit<8> total_hops; // number of hops that inserted INT metadata
            bit<16> instruction_mask;
            bit<16> rsvd2;
        }'''
        
        '''  INT header version 1.0
        header int_header_t {
            bit<4>  ver;
            bit<2>  rep;
            bit<1>  c;
            bit<1>  e;
            bit<1>  m;
            bit<7>  rsvd1;
            bit<3>  rsvd2;
            bit<5>  hop_metadata_len;   // the length of the metadata added by a single INT node (4-byte words)
            bit<8>  remaining_hop_cnt;  // how many switches can still add INT metadata
            bit<16>  instruction_mask;   
            bit<16> rsvd3;
        }'''


        # int header
        self.int_hdr = data[offset + 4:offset + 12]
        self.int_version = self.int_hdr[0] >> 4  # version in INT v0.4 has only 2 bits!
        if self.int_version == 0: # if rep is 0 then it is ok for INT v0.4
            self.hop_count = self.int_hdr[3]
        elif self.int_version == 1:
            self.hop_metadata_len = int(self.int_hdr[2] & 0x1f)
            self.remaining_hop_cnt = self.int_hdr[3]
            self.hop_count = int(self.int_data_len / self.hop_metadata_len)
            logger.debug("hop_metadata_len: %d, int_data_len: %d, remaining_hop_cnt: %d, hop_count: %d" % (
                            self.hop_metadata_len, self.int_data_len, self.remaining_hop_cnt, self.hop_count))
        else:
            logger.error("Unsupported INT version %s - skipping report" % self.int_version)
            raise Exception("Unsupported INT version %s - skipping report" % self.int_version)

        self.ins_map = int.from_bytes(self.int_hdr[4:5], byteorder='big')
        
        logger.debug(hex(self.ins_map))

        # int metadata
        self.int_meta = data[offset + 12:]
        logger.debug("Metadata (%d bytes) is: %s" % (len(self.int_meta), binascii.hexlify(self.int_meta)))
        self.hop_metadata = []
        self.int_meta = io.BytesIO(self.int_meta)
        for i in range(self.hop_count):
            try:
                hop = HopMetadata(self.int_meta, self.ins_map, self.int_version)
                self.int_meta = hop.unread_data()
                self.hop_metadata.append(hop)
            except Exception as e:
                logger.info("Metadata left (%s position) is: %s" % (self.int_meta.tell(), self.int_meta))
                logger.error(e)
                break
                
        logger.debug(vars(self))

    def __str__(self):
        hop_info = ''
        for hop in self.hop_metadata:
            hop_info += str(hop) + '\n'
        flow_tuple = "src: %(src)s, dst: %(dst)s, src_port: %(scrp)s, dst_port: %(dstp)s, protocol: %(protocol)s" % self.flow_id 
        additional_info =  "sw: %s, seq: %s, int version: %s, ins_map: 0x%x, hops: %d" % (
            self.switch_id,
            self.seq_num,
            self.int_version,
            self.ins_map,
            self.hop_count,
        )
        return "\n".join([flow_tuple, additional_info, hop_info])
        


class IntCollector():
    
    def __init__(self, influx, period):
        self.influx = influx
        self.reports = []
        self.last_endpoint_ingress_timestamp = 0 # save last endpoint ingress_timestamp
        self.last_e2e_delay = 0 # save last e2e delay
        self.last_reordering = {}  # save last `reordering` per each monitored flow
        self.last_hop_latency = {} #save last hop_latency per each hop in each monitored flow
        self.period = period # maximum time delay of int report sending to influx
        self.last_hop_egress_timestamp = 0 # save last hop egress_timestamp 
        self.last_send = time.time() # last time when reports were send to influx
        
    def add_report(self, report):
        self.reports.append(report)
        
        reports_cnt = len(self.reports)
        logger.debug('%d reports ready to sent' % reports_cnt)
        # send if many report ready to send or some time passed from last sending
        if reports_cnt > 100 or time.time() - self.last_send > self.period:
            logger.info("Sending %d reports to influx from last %s secs" % (reports_cnt, time.time() - self.last_send))
            self.__send_reports()
            self.last_send = time.time()
            
    def __prepare_e2e_report(self, report, flow_key):
        # e2e report contains information about end-to-end flow delay,         
        try:
            origin_timestamp = report.hop_metadata[-1].ingress_timestamp
            # egress_timestamp of sink node is creasy delayed - use ingress_timestamp instead
            destination_timestamp = report.hop_metadata[0].ingress_timestamp
        except Exception as e:
            origin_timestamp, destination_timestamp = 0, 0
            logger.error("ingress_timestamp in the INT hop is required, %s" % e)
        
        json_report = {
            "measurement": "int_telemetry",
            "tags": report.flow_id,
            'time': int(time.time()*1e9), # use local time because bmv2 clock is a little slower making time drift 
            "fields": {
                "origts": 1.0*origin_timestamp,
                "dstts": 1.0*destination_timestamp,
                "seq": 1.0*report.seq_num,
                "delay": 1.0*(destination_timestamp-origin_timestamp),
                }
        }
        
        # add sink_jitter only if can be calculated (not first packet in the flow)  
        if self.last_e2e_delay != 0:
            json_report["fields"]["sink_jitter"] = abs(1.0*(destination_timestamp-origin_timestamp) - self.last_e2e_delay)
        
        # add reordering only if can be calculated (not first packet in the flow)  
        if flow_key in self.last_reordering:
            json_report["fields"]["reordering"] = 1.0*report.seq_num - self.last_reordering[flow_key] - 1

        if self.last_endpoint_ingress_timestamp != 0:
            json_report["fields"]["IAT"] = 1.0 * destination_timestamp - self.last_endpoint_ingress_timestamp
        self.last_endpoint_ingress_timestamp = 1.0 * destination_timestamp
                        
        # save dstts for purpose of sink_jitter calculation
        self.last_e2e_delay = 1.0*(destination_timestamp-origin_timestamp)
        
        # save dstts for purpose of sink_jitter calculation
        self.last_reordering[flow_key] = report.seq_num
        return json_report
        
        #~ last_hop_delay = report.hop_metadata[-1].ingress_timestamp
        #~ for index, hop in enumerate(reversed(report.hop_metadata)):
            #~ if "hop_latency" in vars(hop):
                #~ json_report["fields"]["latency_%d" % index] = hop.hop_latency
            #~ if "ingress_timestamp" in vars(hop) and index > 0:
                #~ json_report["fields"]["hop_delay_%d" % index] = hop.ingress_timestamp - last_hop_delay
                #~ last_hop_delay = hop.ingress_timestamp
                
    def __prepare_hop_report(self, report, index, hop, flow_key):
        # each INT hop metadata are sent as independed json message to Influx
        tags = copy(report.flow_id)
        tags['hop_index'] = index
        json_report = {
            "measurement": "int_telemetry",
            "tags": tags,
            'time': int(time.time()*1e9), # use local time because bmv2 clock is a little slower making time drift 
            "fields": {}
        }
        
        # combine flow id with hop index 
        flow_hop_key = (*flow_key, index)
        
        # add sink_jitter only if can be calculated (not first packet in the flow)  
        if flow_hop_key in self.last_hop_latency:
            json_report["fields"]["hop_jitter"] = abs(hop.hop_latency - self.last_hop_latency[flow_hop_key])
            
        if "hop_latency" in vars(hop):
            json_report["fields"]["hop_delay"] = hop.hop_latency
            self.last_hop_latency[flow_hop_key] = hop.hop_latency

        if "queue_occupancy" in vars(hop):
            json_report["fields"]["queue_occupancy"] = hop.queue_occupancy
            
        if "ingress_timestamp" in vars(hop):
            if index > 0:
                json_report["fields"]["link_delay"] = hop.ingress_timestamp - self.last_hop_egress_timestamp
            self.last_hop_egress_timestamp = hop.egress_timestamp
            
        json_report["fields"]["modal_type"] = hop.modal_type
        return json_report
        
        
    def __prepare_reports(self, report):
        flow_key = "%(src)s, %(dst)s, %(scrp)s, %(dstp)s, %(protocol)s" % report.flow_id 
        reports = []
        reports.append(self.__prepare_e2e_report(report, flow_key))
        self.last_hop_egress_timestamp = 0
        for index, hop in enumerate(reversed(report.hop_metadata)):
            reports.append(self.__prepare_hop_report(report, index, hop, flow_key))
        return reports
        
        
    def __send_reports(self):
        json_body = []
        for report in self.reports:
            if report.hop_metadata:
                json_body.extend(self.__prepare_reports(report))
            else:
                logger.warning("Empty report metadata: %s" % str(report))
        logger.info("Json body for influx:\n %s" % pprint.pformat(json_body))
        if json_body:
            try:
                self.influx.write_points(json_body)
                self.last_send = time.time()
                logger.info(" %d int reports sent to the influx" % len(json_body))
            except Exception as e:
                logger.exception(e)
        self.reports = [] # clear reports sent

def unpack_int_report(packet):
    report = IntReport(packet)
    logger.info(report)
    return report
            

def influx_client(args):
    if ':' in args.host:
        host, port = args.host.split(':')
    else:
        host = args.host
        port = 8086
    user = 'int'
    password = 'gn4intp4'
    dbname = args.database

    client = InfluxDBClient(host, port, user, password, dbname)
    logger.info("InfluxDBClient(host:%s, port:%s, user:%s, password:%s, dbname:%s):", host, port, user, password, dbname)
    logger.info("Influx client ping response: %s" % client.ping())
    return client
    
    
def start_udp_server(args):
    bufferSize  = 65565
    port = args.int_port
    
    influx = influx_client(args)
    collector = IntCollector(influx, args.period)

    # Create a datagram socket
    sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    logger.info("UDP server up and listening at UDP port: %d" % port)

    # Listen for incoming datagrams
    while(True):
        message, address = sock.recvfrom(bufferSize)
        logger.info("Received INT report (%d bytes) from: %s" % (len(message), str(address)))
        logger.debug(binascii.hexlify(message))
        try:
            report = unpack_int_report(message)
            if report:
                collector.add_report(report)
        except Exception as e:
            logger.exception("Exception during handling the INT report")


def test_hopmetadata():
    ins_map = 0b11001100 << 8
    data = struct.pack("!I", 1)
    data += struct.pack("!HH", 2, 3)
    data += struct.pack("!Q", 11)
    data += struct.pack("!Q", 12)
    meta = HopMetadata(data, ins_map)
    print(meta)


if __name__ == "__main__":
    args = parse_params()
    if args.debug_mode > 0:
        logger.setLevel(logging.DEBUG)
    start_udp_server(args)

# SELECT mean("node_delay")  FROM int_telemetry  WHERE ("src" =~ /^$src$/ AND "dst" =~ /^$dst$/ AND  "node_index" =~ /^$hop$/) AND $timeFilter  GROUP BY time($interval) fill(null)
# SELECT mean("node_delay") FROM "int_udp_policy"."int_telemetry" WHERE ("src" = '10.0.1.1' AND "dst" = '10.0.2.2' AND "hop_number" = '0') AND $timeFilter GROUP BY time($__interval) fill(null)