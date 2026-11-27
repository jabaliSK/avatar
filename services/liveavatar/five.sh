#!/usr/bin/env bash
# Run only on a real five-device reservation, from the patched upstream checkout.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?set exactly five reserved GPU IDs in order}"
: "${WEIGHTS:?absolute directory containing Wan2.2-S2V-14B and Live-Avatar}"
: "${IMAGE:?same reference as the two-GPU run}"
: "${AUDIO:?same full WAV as the two-GPU run}"
: "${OUT:?new absolute output directory}"
branch_dir=$(cd "$(dirname "$0")" && pwd)
[[ "$OUT" = /* && "$WEIGHTS" = /* && "$IMAGE" = /* && "$AUDIO" = /* ]]
[[ "$(git rev-parse HEAD)" = c3c47d031d8bf3247428333c1fe610579c71a551 ]]
git apply --reverse --check "$branch_dir/two-device.patch"
python -c 'import torch; assert torch.cuda.device_count()==5, "reserve exactly 5 GPUs"'
mkdir "$OUT"
export KKA_TPP_METRICS="$OUT/decoded.json" HF_HUB_OFFLINE=1 ENABLE_COMPILE=false
prompt=$(PYTHONPATH="$branch_dir" python -c 'from bench import PROMPT; print(PROMPT)')
python -m torch.distributed.run --standalone --nproc-per-node=5 minimal_inference/s2v_streaming_interact.py \
    --ulysses_size 1 --task s2v-14B --size '704*384' --base_seed 420 \
    --training_config liveavatar/configs/s2v_causal_sft.yaml --offload_model False --convert_model_dtype \
    --prompt "$prompt" --image "$IMAGE" --audio "$AUDIO" --infer_frames 48 --load_lora \
    --lora_path_dmd "$WEIGHTS/Live-Avatar/liveavatar.safetensors" --sample_steps 4 --sample_guide_scale 0 \
    --num_clip 10000 --num_gpus_dit 4 --sample_solver euler --enable_vae_parallel \
    --ckpt_dir "$WEIGHTS/Wan2.2-S2V-14B" --fp8 --save_file "$OUT/full.mp4"
