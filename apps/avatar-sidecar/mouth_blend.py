"""Put MuseTalk's mouth back over a face-restored frame.

GFPGAN redraws the lips too, and on some frames that leaves a doubled or ghosted
mouth. The MuseTalk frame already has the mouth in sync with the speech, so this
keeps MuseTalk's pixels inside a feathered mask around the mouth landmarks
(68-point indices 48-67) and takes everything else from the restored frame.

Frames are RGB uint8 arrays, the same as the rest of the sidecar.
"""

from __future__ import annotations

import cv2
import numpy as np

MOUTH_POINTS = slice(48, 68)
MOUTH_SCALE = 1.35  # grow the hull around the mouth centre so the seam stays off the lips
SHARPEN_AMOUNT = 1.4  # unsharp-mask strength on the MuseTalk mouth (it is redrawn at 256 px and looks soft)
SHARPEN_SIGMA = 1.2


def mouth_mask(landmarks: np.ndarray, height: int, width: int) -> np.ndarray:
    """Soft mask (H, W, 1) in [0, 1] covering the mouth. Zeros if landmarks are missing."""
    mask = np.zeros((height, width), np.float32)
    if landmarks is not None:
        pts = landmarks[MOUTH_POINTS].astype(np.float32)
        centre = pts.mean(axis=0)
        pts = centre + (pts - centre) * MOUTH_SCALE
        hull = cv2.convexHull(pts.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1.0)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=4 + 0.01 * max(height, width))
    return mask[..., None]


def sharpen(frame: np.ndarray, amount: float = SHARPEN_AMOUNT, sigma: float = SHARPEN_SIGMA) -> np.ndarray:
    """Unsharp mask on a float32 frame."""
    blur = cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma)
    return frame + amount * (frame - blur)


def blend_mouth(mouth_frames: np.ndarray, restored_frames: np.ndarray, fa, sharpen_mouth: bool = True) -> np.ndarray:
    """mouth_frames: MuseTalk output [T,H,W,3] RGB; restored_frames: GFPGAN output, same shape.

    `fa` is a face_alignment.FaceAlignment instance (CPU). Returns the blended frames.
    """
    out = np.empty_like(restored_frames)
    for i in range(len(restored_frames)):
        found = fa.get_landmarks(mouth_frames[i])
        lm = found[0] if found else None
        m = mouth_mask(lm, *restored_frames[i].shape[:2])
        mouth = mouth_frames[i].astype(np.float32)
        if sharpen_mouth:
            mouth = sharpen(mouth)
        blended = m * mouth + (1.0 - m) * restored_frames[i].astype(np.float32)
        out[i] = blended.clip(0, 255).astype(np.uint8)
    return out
