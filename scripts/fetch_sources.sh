#!/usr/bin/env bash
# Clone third-party sources at pinned revisions into third_party/. Idempotent.
set -euo pipefail
cd "$(dirname "$0")/../third_party"

clone_pin() {  # name url sha
  local name=$1 url=$2 sha=$3
  if [ ! -d "$name/.git" ]; then
    git clone --quiet --filter=blob:none "$url" "$name"
  fi
  git -C "$name" fetch --quiet origin "$sha" 2>/dev/null || git -C "$name" fetch --quiet origin
  git -C "$name" checkout --quiet --detach "$sha"
  echo "$name @ $(git -C "$name" rev-parse --short=12 HEAD)  ($url)"
}

# Apache-2.0 — verified via GitHub API / LICENSE on 2026-11-25.
clone_pin SoulX-FlashHead https://github.com/Soul-AILab/SoulX-FlashHead.git 9bc03de06bb0de82cd6bc477804512ae06144bf2
clone_pin echomimic_v3    https://github.com/antgroup/echomimic_v3.git       7e89489ca51c0d008fc1963ec6c03fc5bd0b9397
clone_pin OpenAvatarChat  https://github.com/HumanAIGC-Engineering/OpenAvatarChat.git 8b7b3b45bca28ae9ab5fa72ee31bc03c5a99b08b
clone_pin ditto-talkinghead https://github.com/antgroup/ditto-talkinghead.git c3e47eee2e626500017a0556b470d6d4182f85e8
clone_pin grid-sample3d-trt-plugin https://github.com/SeanWangJS/grid-sample3d-trt-plugin.git f964750b8ce8d5453251a4036572d471a4c395e1

# OpenAvatarChat git submodules (pinned to the commits recorded in OAC v0.6.0), placed exactly where OAC expects them.
# Silero VAD: MIT, ships silero_vad.onnx.  OpenAvatarChat-WebUI: Apache-2.0, ships a prebuilt dist/.
OAC=OpenAvatarChat
clone_pin "$OAC/src/handlers/vad/silerovad/silero_vad"     https://github.com/snakers4/silero-vad.git                                9060f664f20eabb66328e4002a41479ff288f14c
clone_pin "$OAC/src/service/frontend_service/frontend"      https://github.com/HumanAIGC-Engineering/OpenAvatarChat-WebUI.git       a6182afbda3f3b84a6608402a41e55a0c7bc6766
