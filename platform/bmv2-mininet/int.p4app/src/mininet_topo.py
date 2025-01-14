import subprocess
import time
from mininet.topo import Topo

_THRIFT_BASE_PORT = 22222

def float_to_custom_bin(number):
    # 确定符号位并取绝对值
    if number < 0:
        sign_bit = '01'
        number = -number
    else:
        sign_bit = '00'

    # 分离整数部分和小数部分
    integer_part, fractional_part = divmod(number, 1)
    integer_part = int(integer_part)

    # 将整数部分转换为 15 位二进制
    integer_bits = format(integer_part, '015b')

    # 将小数部分转换为 15 位二进制
    fractional_bits = ''
    while len(fractional_bits) < 15:
        fractional_part *= 2
        bit, fractional_part = divmod(fractional_part, 1)
        fractional_bits += str(int(bit))

    # 拼接符号位、整数二进制和小数二进制
    binary_representation = sign_bit + integer_bits + fractional_bits
    decimal_representation = int(binary_representation, 2)
    return decimal_representation

def customFlexIP(vmx, i):
    F0 = 2048 + vmx * 100 + i - 64
    F1 = 7264804 + vmx * 1000 + i - 64
    F2 = 13849245 + vmx * 10000 + i - 64
    F4 = 202271720 + vmx * 100000 + i - 64
    if 0<=i<=3:
        return "F0/{:04X}".format(F0)
    elif 4<=i<=7:
        return "F1/{:08X}".format(F1)
    elif 8<=i<=11:
        return "F2/{:016X}".format(F2)
    elif 12<=i<=15:
        return "F4/{:064X}".format(F4)
    elif i==16:
        return "F6/{:02X}/F0/{:04X}".format(i, F0)
    elif i==17:
        return "F6/{:02X}/F1/{:08X}".format(i, F1)
    elif i==18:
        return "F6/{:02X}/F2/{:016X}".format(i, F2)
    elif i==19:
        return "F6/{:02X}/F4/{:064X}".format(i, F4)
    elif i==20:
        return "F6/F0/{:04X}/F1/{:08X}".format(F0, F1)
    elif i==21:
        return "F6/F0/{:04X}/F2/{:016X}".format(F0, F2)
    elif i==22:
        return "F6/F0/{:04X}/F4/{:064X}".format(F0, F4)
    elif i==23:
        return "F6/F1/{:08X}/F2/{:016X}".format(F1, F2)
    elif i==24:
        return "F6/F1/{:08X}/F4/{:064X}".format(F1, F4)
    elif i==25:
        return "F6/F2/{:016X}/F4/{:064X}".format(F2, F4)
    elif i==26:
        return "F7/{:02X}/F0/{:04X}/F1/{:08X}".format(i, F0, F1)
    elif i==27:
        return "F7/{:02X}/F0/{:04X}/F2/{:016X}".format(i, F0, F2)
    elif i==28:
        return "F7/{:02X}/F0/{:04X}/F4/{:064X}".format(i, F0, F4)
    elif i==29:
        return "F7/{:02X}/F1/{:08X}/F2/{:016X}".format(i, F1, F2)
    elif i==30:
        return "F7/{:02X}/F1/{:08X}/F4/{:064X}".format(i, F1, F4)
    elif i==31:
        return "F7/{:02X}/F2/{:016X}/F4/{:064X}".format(i, F2, F4)
    elif i==32:
        return "F7/F0/{:04X}/F1/{:08X}/F2/{:016X}".format(F0, F1, F2)
    elif i==33:
        return "F7/F0/{:04X}/F1/{:08X}/F4/{:064X}".format(F0, F1, F4)
    elif i==34:
        return "F7/F0/{:04X}/F2/{:016X}/F4/{:064X}".format(F0, F2, F4)
    elif i==35:
        return "F7/F1/{:08X}/F2/{:016X}/F4/{:064X}".format(F1, F2, F4)
    return "{:02X}".format(i)

class ModalHost(Host):
    def __init__(self, name, inNamespace=True, **params):
        Host.__init__(self, name, inNamespace=inNamespace, **params)

    def config(self, identity=None, mf_guid=None, geoPosLat=None, geoPosLon=None, disa=None, disb=None, ndn_name=None, ndn_content=None, flexip=None, **params):
        r = super(Host, self).config(**params)
        for off in ["rx", "tx", "sg"]:
            cmd = "/sbin/ethtool --offload %s %s off" \
                  % (self.defaultIntf(), off)
            self.cmd(cmd)
        self.identity = identity
        self.mf_guid = mf_guid
        self.geoPosLat = geoPosLat
        self.geoPosLon = geoPosLon
        self.disa = disa
        self.disb = disb
        self.ndn_name = ndn_name
        self.ndn_content = ndn_content
        self.flexip = flexip
        # disable IPv6
        self.cmd("sysctl -w net.ipv6.conf.all.disable_ipv6=1")
        self.cmd("sysctl -w net.ipv6.conf.default.disable_ipv6=1")
        self.cmd("sysctl -w net.ipv6.conf.lo.disable_ipv6=1")

        # 设置默认路由
        self.cmd("ip route add default via 218.199.84.161")
        return r

    def ID(self):
        return self.identity
    
    def MF(self):
        return self.mf_guid

    def FlexIP(self):
        return self.flexip

class MyTopo(Topo):
    def __init__(self, sw_path, json_path, nb_hosts, nb_switches, links, **opts):
        # Initialize topology and default options
        Topo.__init__(self, **opts)
        for i in range(nb_switches):
           self.addSwitch('s%d' % (i + 1),
                            sw_path = sw_path,
                            json_path = json_path,
                            thrift_port = _THRIFT_BASE_PORT + i,
                            pcap_dump = False,
                            device_id = i,
                            enable_debugger = True)
        vmx = 0
        for i in range(nb_hosts):
            self.addHost('h%d' % (i + 1),
                    cls=ModalHost,
                    ip="10.0.%d.%d" % ((i + 1) , (i + 1)),
                    mac="00:00:00:00:0%d:0%d" % ((i+1), (i+1)),
                    identity=202271720 + vmx * 100000 + i,
                    mf_guid=1 + vmx * 1000 + i,
                    geoPosLat=i,
                    geoPosLon=float_to_custom_bin(-180 + vmx * 20 + i * 0.4),
                    disa=0,
                    disb=0,
                    ndn_name=13849245 + vmx * 100000 + i,
                    ndn_content=2048 + vmx * 1000 + i,
                    flexip=customFlexIP(vmx, i))

        for a, b in links:
            self.addLink(a, b)
            

def read_topo():
    nb_hosts = 0
    nb_switches = 0
    links = []
    with open("topo.txt", "r") as f:
        line = f.readline()[:-1]
        w, nb_switches = line.split()
        assert(w == "switches")
        line = f.readline()[:-1]
        w, nb_hosts = line.split()
        assert(w == "hosts")
        for line in f:
            if not f: break
            a, b = line.split()
            links.append( (a, b) )
    return int(nb_hosts), int(nb_switches), links



       
def configure_hosts(net, nb_hosts):
    for n in range(nb_hosts):
        h = net.get('h%d' % (n + 1))
        for off in ["rx", "tx", "sg"]:
            cmd = "/sbin/ethtool --offload eth0 %s off" % off
            print(cmd)
            h.cmd(cmd)
        print("disable ipv6")
        h.cmd("sysctl -w net.ipv6.conf.all.disable_ipv6=1")
        h.cmd("sysctl -w net.ipv6.conf.default.disable_ipv6=1")
        h.cmd("sysctl -w net.ipv6.conf.lo.disable_ipv6=1")
        h.cmd("sysctl -w net.ipv4.tcp_congestion_control=reno")
        h.cmd("iptables -I OUTPUT -p icmp --icmp-type destination-unreachable -j DROP")
    time.sleep(1)


def configure_switches(net, nb_switches, args):    
    for i in range(nb_switches):
        cmd = [args.cli, "--json", args.json,
               "--thrift-port", str(_THRIFT_BASE_PORT + i)
               ]
        with open("commands/commands"+str((i+1))+".txt", "r") as f:
            print(" ".join(cmd))
            try:
                output = subprocess.check_output(cmd, stdin = f)
                print(output.decode('ascii'))
            except subprocess.CalledProcessError as e:
                print(e)
                print(e.output)

        s = net.get('s%d' % (i + 1))
        s.cmd("sysctl -w net.ipv6.conf.all.disable_ipv6=1")
        s.cmd("sysctl -w net.ipv6.conf.default.disable_ipv6=1")
        s.cmd("sysctl -w net.ipv6.conf.lo.disable_ipv6=1")
        s.cmd("sysctl -w net.ipv4.tcp_congestion_control=reno")
        s.cmd("iptables -I OUTPUT -p icmp --icmp-type destination-unreachable -j DROP")
