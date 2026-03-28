"""
Generate test_input.json for local RunPod handler testing.

Loads the first sample from data/val.h5 and serialises it into the
handler's expected input format.

Usage:
    python make_test_input.py [--h5 data/val.h5] [--idx 0] [--out test_input.json]
"""

import argparse
import base64
import json

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5",  default="data/val.h5")
    ap.add_argument("--idx", type=int, default=0, help="Sample index in the HDF5 file")
    ap.add_argument("--out", default="test_input.json")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as hf:
        keys = sorted(hf.keys())
        key  = keys[args.idx]
        point_seq       = np.array(hf[key]["point_seq"][:], dtype=np.float32)
        char_points_idx = hf[key]["char_points_idx"][:].tolist()
        line_text       = hf[key]["line_text"][()].decode("utf-8")

    strokes_b64 = base64.b64encode(point_seq.tobytes()).decode("utf-8")

    payload = {
        "input": {
            "text": line_text,
            "style_strokes": strokes_b64,
            "char_points_idx": char_points_idx,
            "sampling_timesteps": 20,
            "cfg_scale": 1.0,
            "temperature": 0.1,
            "greedy": True,
            "output_image": True,
        }
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.out}")
    print(f"  key:   {key}")
    print(f"  text:  {line_text}")
    print(f"  shape: {point_seq.shape}")
    print(f"  chars: {len(char_points_idx)}")


if __name__ == "__main__":
    main()
