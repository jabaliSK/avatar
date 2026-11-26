#!/bin/sh
# Gateway entrypoint: optional self-signed TLS (browsers only allow the microphone on https or localhost), the default avatar, the
# TURN relay for browsers on other machines, then OpenAvatarChat through launch.py.
set -eu

TLS_DIR=/opt/kka/tls
if [ "${GATEWAY_TLS:-0}" = "1" ] && [ ! -s "$TLS_DIR/cert.pem" ]; then
    mkdir -p "$TLS_DIR"
    # Generated at start so no key material is ever baked into an image. Use a real certificate for anything but a demo.
    SAN="DNS:localhost,IP:127.0.0.1"
    [ -n "${GATEWAY_TLS_SAN:-}" ] && SAN="$SAN,$GATEWAY_TLS_SAN"
    openssl req -x509 -newkey rsa:2048 -nodes -days 365 -keyout "$TLS_DIR/key.pem" -out "$TLS_DIR/cert.pem" \
        -subj "/CN=${GATEWAY_TLS_CN:-localhost}" -addext "subjectAltName=$SAN" 2>/dev/null
    echo "gateway: generated a self-signed certificate (SAN: $SAN)"
fi

# The avatar handler reads the avatar name from this file at the start of every session, so it can be switched without a restart.
mkdir -p /run/kka
if [ -n "${KKA_AVATAR:-}" ] && [ ! -s /run/kka/avatar ]; then
    echo "$KKA_AVATAR" > /run/kka/avatar
fi

# With KKA_TURN_HOST the browsers are given a TURN relay (see compose.yaml, service coturn). Its address and credentials are not in the
# image, so the config in use is written here. OpenAvatarChat lets OPEN_AVATAR_CHAT_CONFIG override --config, so both are set.
CONFIG="${OPEN_AVATAR_CHAT_CONFIG:-/opt/oac/config/kka.yaml}"
if [ -n "${KKA_TURN_HOST:-}" ]; then
    python /opt/kka/render_config.py "$CONFIG" /tmp/kka.yaml
    CONFIG=/tmp/kka.yaml
fi
export OPEN_AVATAR_CHAT_CONFIG="$CONFIG"

cd /opt/oac
exec python /opt/kka/launch.py --config "$CONFIG" "$@"
