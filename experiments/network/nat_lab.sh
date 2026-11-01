#!/bin/bash
# A small internet on one machine, for C7: a "public" bridge and two endpoints, each
# behind its own NAT namespace. Usage: nat_lab.sh up SELLER_NAT BUYER_NAT [noinputdrop] | down
# NAT types: full (full cone), cone (port-restricted, keeps the port), sym (symmetric),
# rport (port-restricted, one fixed public port that is not the local one).
# Like a home router, a NAT drops what reaches its WAN side unasked. `noinputdrop` leaves
# that out: the packet then gets as far as the router's own stack, and conntrack keeps
# an entry for it, which is what the first C7 run measured by accident.
set -e
down() {
    set +e
    for ns in mxeps mxepb mxepp; do ip netns pids $ns 2>/dev/null | xargs -r kill -9; done
    for ns in mxnats mxnatb mxeps mxepb mxepp; do ip netns del $ns 2>/dev/null; done
    ip link del mxbr 2>/dev/null
    while iptables -D INPUT -i mxbr -j ACCEPT 2>/dev/null; do :; done
    while iptables -D FORWARD -i mxbr -o mxbr -j ACCEPT 2>/dev/null; do :; done
    return 0
}

nat_rules() {  # ns, wan ip, endpoint ip, type
    if [ "$INPUT_DROP" = yes ]; then
        ip netns exec $1 iptables -A INPUT -i wan -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
        ip netns exec $1 iptables -A INPUT -i wan -j DROP
    fi
    case $4 in
    full) ip netns exec $1 iptables -t nat -A POSTROUTING -o wan -j SNAT --to-source $2
          ip netns exec $1 iptables -t nat -A PREROUTING -i wan -j DNAT --to-destination $3 ;;
    cone) ip netns exec $1 iptables -t nat -A POSTROUTING -o wan -j SNAT --to-source $2 ;;
    sym)  ip netns exec $1 iptables -t nat -A POSTROUTING -o wan -j MASQUERADE --random ;;
    rport) local base=$((RANDOM % 30000 + 20000))  # a real NAT maps each local port on its own
           ip netns exec $1 iptables -t nat -A POSTROUTING -o wan -p udp --sport 41781 -j SNAT --to-source $2:$((base + 1))
           ip netns exec $1 iptables -t nat -A POSTROUTING -o wan -p udp -j SNAT --to-source $2:$base
           ip netns exec $1 iptables -t nat -A POSTROUTING -o wan -j SNAT --to-source $2 ;;
    *) echo "unknown NAT $4"; exit 2 ;;
    esac
}

side() {  # letter, number, type
    local nat=mxnat$1 ep=mxep$1 wan=10.77.0.$2 lan=10.77.$2.1 host=10.77.$2.2
    ip netns add $nat
    ip netns add $ep
    ip link add mxw$1 type veth peer name wan netns $nat
    ip link set mxw$1 master mxbr up
    ip -n $nat addr add $wan/24 dev wan
    ip -n $nat link set wan up
    ip -n $nat link set lo up
    ip link add mxl$1 netns $nat type veth peer name eth0 netns $ep
    ip -n $nat link set mxl$1 name lan
    ip -n $nat addr add $lan/24 dev lan
    ip -n $nat link set lan up
    ip -n $ep addr add $host/24 dev eth0
    ip -n $ep link set eth0 up
    ip -n $ep link set lo up
    ip -n $ep route add default via $lan
    ip netns exec $nat sysctl -qw net.ipv4.ip_forward=1
    nat_rules $nat $wan $host $3
}

public() {  # a machine with no NAT at all
    ip netns add mxepp
    ip link add mxwp type veth peer name eth0 netns mxepp
    ip link set mxwp master mxbr up
    ip -n mxepp addr add 10.77.0.9/24 dev eth0
    ip -n mxepp link set eth0 up
    ip -n mxepp link set lo up
}

case $1 in
up)
    INPUT_DROP=yes
    [ "${4:-}" = noinputdrop ] && INPUT_DROP=no
    down
    ip link add mxbr type bridge
    ip addr add 10.77.0.1/24 dev mxbr
    ip link set mxbr up
    iptables -I INPUT -i mxbr -j ACCEPT
    iptables -I FORWARD -i mxbr -o mxbr -j ACCEPT
    side s 2 $2
    side b 3 $3
    public ;;
down) down ;;
esac
