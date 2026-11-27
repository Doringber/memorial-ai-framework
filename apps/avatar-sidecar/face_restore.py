"""Face restoration pass with GFPGAN v1.4, run as a worker process.

GFPGAN needs torch + basicsr, which conflict with the sidecar's MuseTalk/MLX
stack, so the sidecar calls this file with a separate interpreter:

    <gfpgan-venv>/bin/python face_restore.py IN_DIR OUT_DIR --weights GFPGANv1.4.pth

IN_DIR holds numbered PNG frames (BGR, as OpenCV writes them); OUT_DIR gets the
restored frames with the same names. Run it with the working directory set to a
folder you own: facexlib writes its detector weights into the working directory.

GFPGAN is Apache-2.0. Weights: TencentARC/GFPGAN releases (GFPGANv1.4.pth).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2


def main() -> None:
    ap = argparse.ArgumentParser(description="GFPGAN face restoration over a folder of PNG frames")
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--weights", required=True, help="path to GFPGANv1.4.pth")
    ap.add_argument("--blend", type=float, default=0.5, help="restoration strength, 0..1")
    args = ap.parse_args()

    from gfpgan import GFPGANer

    restorer = GFPGANer(
        model_path=args.weights, upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None
    )
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.in_dir, "*.png")))
    for i, path in enumerate(files):
        img = cv2.imread(path)
        _, _, out = restorer.enhance(
            img, has_aligned=False, only_center_face=True, paste_back=True, weight=args.blend
        )
        # No face found: keep the frame as it was rather than dropping it.
        cv2.imwrite(os.path.join(args.out_dir, os.path.basename(path)), out if out is not None else img)
        if i % 50 == 0:
            print(f"restore {i}/{len(files)}", file=sys.stderr, flush=True)
    print(f"restore done {len(files)}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
