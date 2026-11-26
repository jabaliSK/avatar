#!/usr/bin/env bash
# Refuses to continue unless GPU_ID in .env is the physical GPU named by GPU_PCI_BUS in .env.
# This host has two GPUs, so a re-ordered device list must never move this stack to the other one:
#   scripts/gpu_guard.sh && docker compose up -d
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || { echo "gpu_guard: .env not found (copy .env.example)" >&2; exit 2; }
set -a; . ./.env; set +a
: "${GPU_ID:?GPU_ID is not set in .env}"
: "${GPU_PCI_BUS:?GPU_PCI_BUS is not set in .env}"

actual="$(nvidia-smi -i "$GPU_ID" --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | tr -d '[:space:]')"
if [ -z "$actual" ]; then
    echo "gpu_guard: nvidia-smi cannot see GPU $GPU_ID" >&2
    exit 1
fi
if [ "${actual,,}" != "$(printf '%s' "$GPU_PCI_BUS" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')" ]; then
    echo "gpu_guard: GPU_ID=$GPU_ID is at PCI $actual but .env expects $GPU_PCI_BUS. Fix GPU_ID before starting anything." >&2
    exit 1
fi
echo "gpu_guard: GPU $GPU_ID is $actual ($(nvidia-smi -i "$GPU_ID" --query-gpu=name --format=csv,noheader))"
