"""Write the gateway configuration used at run time from the baked kka.yaml.

A TURN server cannot be baked into the image: its address and credentials belong to the deployment. When KKA_TURN_HOST is set, the
RtcClient handler gets a `turn_config`, which OpenAvatarChat hands to the browser as its ICE servers. Without it the file is used as is.

    python render_config.py <source.yaml> <target.yaml>
"""
import copy
import os
import re
import sys

import yaml

# These values end up in a URL, in the coturn config file and in Dynaconf, which treats a leading "@" specially.
SAFE = re.compile(r"^[A-Za-z0-9._~-]+$")


def turn_config(env) -> dict | None:
    host = env.get("KKA_TURN_HOST", "").strip()
    if not host:
        return None
    user = env.get("KKA_TURN_USER", "").strip()
    password = env.get("KKA_TURN_PASSWORD", "").strip()
    port = env.get("KKA_TURN_PORT", "3478").strip()
    if not user or not password:
        raise ValueError("KKA_TURN_HOST is set, so KKA_TURN_USER and KKA_TURN_PASSWORD must be set as well")
    for name, value in (("KKA_TURN_HOST", host), ("KKA_TURN_USER", user), ("KKA_TURN_PASSWORD", password)):
        if not SAFE.match(value):
            raise ValueError(f"{name} may only contain letters, digits and . _ ~ -")
    if not port.isdigit():
        raise ValueError("KKA_TURN_PORT must be a port number")
    return {
        "turn_provider": "turn_server",
        # UDP first; TCP is the way in for a client whose network blocks UDP to the relay (the relayed media is UDP either way).
        "urls": [f"turn:{host}:{port}?transport=udp", f"turn:{host}:{port}?transport=tcp"],
        "username": user,
        "credential": password,
    }


def render(config: dict, env) -> dict:
    turn = turn_config(env)
    if turn is None:
        return config
    out = copy.deepcopy(config)
    found = 0
    for section in out.values():  # one section per environment ("default")
        handlers = ((section or {}).get("chat_engine") or {}).get("handler_configs") or {}
        if "RtcClient" in handlers:
            handlers["RtcClient"]["turn_config"] = turn
            found += 1
    if not found:
        raise ValueError("no RtcClient handler in the config, so there is nowhere to put the TURN server")
    return out


def main(argv) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    with open(argv[1]) as f:
        config = yaml.safe_load(f)
    try:
        config = render(config, os.environ)
    except ValueError as e:
        print(f"gateway: {e}", file=sys.stderr)
        return 1
    with open(argv[2], "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False, width=200, allow_unicode=True)
    turn = turn_config(os.environ)
    if turn:
        print(f"gateway: browsers get the TURN relay {turn['urls'][0].split('?')[0]} (user {turn['username']})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
