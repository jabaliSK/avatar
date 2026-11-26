#!/bin/sh
# TURN relay entrypoint (coturn, BSD-3-Clause): builds turnserver.conf from the environment and starts turnserver.
#
# The gateway's WebRTC media is UDP to its container address (172.x), which a browser on another machine cannot reach. The browser
# sends its media to this relay instead (published on the host as 3478, UDP and TCP) and coturn passes it on to the gateway over the
# compose network. The relay ports 49160-49199 stay inside the container: only the gateway talks to them, and it answers the relay's
# private address.
set -eu

: "${TURN_EXTERNAL_IP:?TURN_HOST is not set in .env (the address clients use to reach this host)}"
: "${TURN_USER:?TURN_USER is not set in .env}"
: "${TURN_PASSWORD:?TURN_PASSWORD is not set in .env}"

# The credentials go into a config file and into a URL: keep them to characters that need no escaping in either.
case "$TURN_USER$TURN_PASSWORD" in
    *[!A-Za-z0-9._~-]*) echo "coturn: TURN_USER and TURN_PASSWORD may only contain letters, digits and . _ ~ -" >&2; exit 1 ;;
esac

# coturn wants the public address as an IP.
case "$TURN_EXTERNAL_IP" in
    *[!0-9.]*) IP=$(getent ahostsv4 "$TURN_EXTERNAL_IP" | awk 'NR == 1 { print $1 }') ;;
    *) IP=$TURN_EXTERNAL_IP ;;
esac
[ -n "$IP" ] || { echo "coturn: cannot resolve $TURN_EXTERNAL_IP" >&2; exit 1; }

# Relay only to the network this container is on, which is where the gateway is. Without this, anyone holding the credentials could
# use the relay to send UDP to any host the container can route to. The range comes from the directly connected routes in
# /proc/net/route, whose addresses are little-endian hex (172.18.0.0/16 is 000012AC with mask 0000FFFF).
PEERS=$(awk '
function hex(s,   i, n) { n = 0; for (i = 1; i <= length(s); i++) n = n * 16 + index("0123456789ABCDEF", substr(s, i, 1)) - 1; return n }
function octet(s, k) { return hex(substr(s, 9 - 2 * k, 2)) }
NR > 1 && $3 == "00000000" && $2 != "00000000" && $8 != "00000000" {
    lo = ""; hi = ""
    for (k = 1; k <= 4; k++) { n = octet($2, k); m = octet($8, k); lo = lo (k > 1 ? "." : "") n; hi = hi (k > 1 ? "." : "") (n + 255 - m) }
    print lo "-" hi
}' /proc/net/route)
[ -n "$PEERS" ] || { echo "coturn: cannot work out this container's network from /proc/net/route" >&2; exit 1; }

CONF=/tmp/turnserver.conf
# One IPv4 listener and relay (the container's own address); without these coturn also listens and relays on loopback and ::1.
LOCAL=$(hostname -i | awk '{ print $1 }')
case "$LOCAL" in ""|*[!0-9.]*) echo "coturn: cannot find this container's IPv4 address (got '$LOCAL')" >&2; exit 1 ;; esac
{
    echo "listening-port=3478"
    echo "listening-ip=$LOCAL"
    echo "relay-ip=$LOCAL"
    echo "external-ip=$IP"
    echo "min-port=49160"
    echo "max-port=49199"
    echo "realm=kka"
    echo "lt-cred-mech"
    echo "user=$TURN_USER:$TURN_PASSWORD"
    echo "fingerprint"
    echo "no-tls"
    echo "no-tcp-relay"
    echo "no-multicast-peers"
    echo "denied-peer-ip=0.0.0.0-255.255.255.255"
    for range in $PEERS; do echo "allowed-peer-ip=$range"; done
    echo "log-file=stdout"
} > "$CONF"

echo "coturn: relaying for $IP, ports 49160-49199, peers allowed only in: $(echo $PEERS)"
exec turnserver -c "$CONF"
