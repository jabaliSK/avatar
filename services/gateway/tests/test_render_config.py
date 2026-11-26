"""render_config: the TURN relay is added to the gateway config only when asked for, and bad values are refused (no GPU)."""
import copy
import os
import sys

import pytest
import yaml

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

import render_config as rc  # noqa: E402

BASE = {"default": {"service": {"port": 8282}, "chat_engine": {"handler_configs": {
    "RtcClient": {"module": "client/rtc_client/client_handler_rtc", "output_video_fps": 25},
    "BargeInVad": {"module": "kka/vad/barge_in_vad", "end_delay": 11200}}}}}
ENV = {"KKA_TURN_HOST": "10.25.73.101", "KKA_TURN_USER": "kka", "KKA_TURN_PASSWORD": "Ab1-_.~x"}


def test_no_turn_host_leaves_the_config_alone():
    assert rc.render(BASE, {}) is BASE
    assert rc.render(BASE, {"KKA_TURN_HOST": "  "}) is BASE


def test_turn_host_adds_turn_config_to_the_rtc_client_only():
    before = copy.deepcopy(BASE)
    out = rc.render(BASE, ENV)
    handlers = out["default"]["chat_engine"]["handler_configs"]
    assert handlers["RtcClient"]["turn_config"] == {
        "turn_provider": "turn_server",
        "urls": ["turn:10.25.73.101:3478?transport=udp", "turn:10.25.73.101:3478?transport=tcp"],
        "username": "kka",
        "credential": "Ab1-_.~x",
    }
    assert handlers["BargeInVad"] == BASE["default"]["chat_engine"]["handler_configs"]["BargeInVad"]
    assert BASE == before  # the input is not modified


def test_custom_port():
    out = rc.render(BASE, {**ENV, "KKA_TURN_PORT": "5349"})
    assert out["default"]["chat_engine"]["handler_configs"]["RtcClient"]["turn_config"]["urls"][0] == "turn:10.25.73.101:5349?transport=udp"


@pytest.mark.parametrize("drop", ["KKA_TURN_USER", "KKA_TURN_PASSWORD"])
def test_credentials_are_required(drop):
    with pytest.raises(ValueError, match="must be set"):
        rc.render(BASE, {k: v for k, v in ENV.items() if k != drop})


@pytest.mark.parametrize("name,value", [
    ("KKA_TURN_PASSWORD", "@json {}"),       # Dynaconf would evaluate it
    ("KKA_TURN_PASSWORD", "pa ss"),
    ("KKA_TURN_USER", "kka:x"),
    ("KKA_TURN_HOST", "10.0.0.1/evil"),
    ("KKA_TURN_PORT", "34a8"),
])
def test_unsafe_values_are_refused(name, value):
    with pytest.raises(ValueError):
        rc.render(BASE, {**ENV, name: value})


def test_no_rtc_client_is_an_error():
    with pytest.raises(ValueError, match="RtcClient"):
        rc.render({"default": {"chat_engine": {"handler_configs": {}}}}, ENV)


def test_shipped_config_survives_the_round_trip_through_dynaconf(tmp_path, monkeypatch):
    dynaconf = pytest.importorskip("dynaconf")
    src = os.path.join(HERE, "..", "config", "kka.yaml")
    dst = tmp_path / "kka.yaml"
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    assert rc.main(["render_config.py", src, str(dst)]) == 0

    monkeypatch.setenv("ENV_FOR_DYNACONF", "default")

    def load(path):
        return dynaconf.Dynaconf(settings_files=[str(path)], environments=True, load_dotenv=False).get("chat_engine")["handler_configs"]

    baked, rendered = load(src), load(dst)
    turn = rendered["RtcClient"].pop("turn_config")
    assert turn["username"] == "kka" and turn["urls"][0].startswith("turn:10.25.73.101:3478")
    assert rendered == baked  # nothing else changed, strings and numbers included
    assert yaml.safe_load(open(dst))["default"]["service"] == yaml.safe_load(open(src))["default"]["service"]
