"""Stand-in for generate_loops.py in the worker tests: the same arguments and log lines, no GPU.

FAKE_MODE: ok | oom (the speaking clip runs out of memory) | crash (exit code 3 after loading) | slow (the speaking clip never finishes)
FAKE_ARGS_LOG: file that gets this run's arguments appended as one JSON line
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--job", action="append", required=True)
ap.add_argument("--subject", action="append", default=[])
ap.add_argument("--states")
ap.add_argument("--speech")
ap.add_argument("--sample-area", type=int)
ap.add_argument("--end-frame")
ap.add_argument("--out")
args = ap.parse_args()
mode = os.environ.get("FAKE_MODE", "ok")
name, image = args.job[0].split("=", 1)
with open(os.environ["FAKE_ARGS_LOG"], "a") as f:
    f.write(json.dumps({**vars(args), "mode": mode}) + "\n")

STAMP = "2026-01-01 00:00:00,000"
print("Namespace(" + "parameter=1, " * 80 + ")", flush=True)  # the real one starts with a huge parameter dump
print("Loading checkpoint shards: 100%|##########| 3/3 [00:00<00:00, 9.9it/s]", flush=True)
print(f"{STAMP} INFO kka.bodygen: models ready in 0.1s (GPU memory 3.0 GB)", flush=True)
if mode == "crash":
    sys.exit(3)
os.makedirs(os.path.join(args.out, name), exist_ok=True)
for state in args.states.split(","):
    print(f"{STAMP} INFO kka.bodygen: {name}/{state}: generating (8 steps, image 64x48, seed 1, end frame pinned)", flush=True)
    for step in range(1, 9):  # tqdm redraws one line with carriage returns
        sys.stdout.write(f"\r {step * 100 // 8:3d}%|#####     | {step}/8 [00:0{step}<00:00, 1.00s/it]")
        sys.stdout.flush()
        time.sleep(0.01)
    sys.stdout.write("\n")
    base = os.path.join(args.out, name, state)
    if mode == "oom" and state == "speaking":
        print(f"{STAMP} ERROR kka.bodygen: {name}/{state}: out of GPU memory; lower --sample-area", flush=True)
        continue
    with open(base + ".mp4", "wb") as f:
        f.write(b"clip")
    if mode == "slow" and state == "speaking":
        time.sleep(60)  # the test cancels the job: the mp4 stays without its report
    with open(base + ".json", "w") as f:
        json.dump({"avatar": name, "state": state}, f)
    print(f"{STAMP} INFO kka.bodygen: {name}/{state}: 3 s, peak 25.9 GB, {{'first_last_mae': 1.0}}", flush=True)
