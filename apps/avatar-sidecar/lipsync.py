"""Second-pass lip sync with MuseTalk 1.5, adapted for Apple Silicon.

AvatarForcing's audio encoder is wav2vec2 trained on English only, so on Hebrew
(and other languages) the mouth only loosely follows the speech. MuseTalk reads
the audio with Whisper (multilingual) and redraws just the lower half of the face
in each frame, so the head motion, blinks and identity from AvatarForcing are
kept and the mouth is re-synced to the actual words.

Differences from upstream scripts/inference.py:
- Face box from face-alignment's 68 landmarks instead of DWPose/mmpose (same
  iBUG-68 point order), so no mmcv/mmpose build is needed on macOS.
- Runs the VAE, UNet and Whisper on MPS; face parsing stays on CPU.
- Frames come in and go out as RGB arrays, no PNG round trip.

MuseTalk code and weights: MIT / free for any use (see its README). The code is
not vendored; point MUSETALK_HOME at a clone with the weights in models/.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

log = logging.getLogger("avatar-sidecar.lipsync")

# v1.5 defaults from upstream scripts/inference.py
EXTRA_MARGIN = 10
AUDIO_PAD_LEFT = 2
AUDIO_PAD_RIGHT = 2
PARSING_MODE = "jaw"
CHEEK_WIDTH = 90


@contextlib.contextmanager
def _cwd(path: Path):
    """MuseTalk's face parser loads its weights from paths relative to the repo root."""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class MuseTalkLipSync:
    def __init__(self, root: str | Path, device: str = "mps", batch_size: int = 16, landmark_stride: int = 5) -> None:
        import face_alignment
        import torch
        from transformers import WhisperModel

        self.root = Path(root).expanduser()
        if not (self.root / "models" / "musetalkV15" / "unet.pth").is_file():
            raise FileNotFoundError(f"MuseTalk weights not found under {self.root}/models (see README)")
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))

        from musetalk.models.unet import PositionalEncoding, UNet
        from musetalk.models.vae import VAE
        from musetalk.utils.audio_processor import AudioProcessor
        from musetalk.utils.blending import get_image
        from musetalk.utils.face_parsing import FaceParsing

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.landmark_stride = max(1, landmark_stride)
        self._get_image = get_image

        models = self.root / "models"
        self.vae = VAE(model_path=str(models / "sd-vae"))
        self.vae.vae.to(self.device)
        self.unet = UNet(
            unet_config=str(models / "musetalkV15" / "musetalk.json"),
            model_path=str(models / "musetalkV15" / "unet.pth"),
            device=torch.device("cpu"),
        )
        self.unet.model.to(self.device).eval()
        self.pe = PositionalEncoding(d_model=384).to(self.device)
        self.audio = AudioProcessor(feature_extractor_path=str(models / "whisper"))
        self.whisper = WhisperModel.from_pretrained(str(models / "whisper")).to(self.device).eval()
        self.whisper.requires_grad_(False)
        with _cwd(self.root):
            self.parser = FaceParsing(left_cheek_width=CHEEK_WIDTH, right_cheek_width=CHEEK_WIDTH)
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device="cpu")
        self.timesteps = torch.tensor([0], device=self.device)

    # ---------------- face box ----------------

    @staticmethod
    def _box_from_landmarks(lm: np.ndarray, height: int) -> tuple[int, int, int, int]:
        """Upstream get_landmark_and_bbox (bbox_shift=0) + v1.5 extra margin."""
        lm = lm.astype(np.int32)
        mid_y = lm[29][1]
        half = np.max(lm[:, 1]) - mid_y
        y1 = max(0, int(mid_y - half))
        x1, x2, y2 = int(np.min(lm[:, 0])), int(np.max(lm[:, 0])), int(np.max(lm[:, 1]))
        return max(0, x1), y1, x2, min(height, y2 + EXTRA_MARGIN)

    def _boxes(self, frames: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Landmarks every `landmark_stride` frames, linearly interpolated in between
        (the avatar's head moves smoothly; full per-frame detection costs ~10x more)."""
        n, h = len(frames), frames.shape[1]
        keys = sorted(set(range(0, n, self.landmark_stride)) | {n - 1})
        known: dict[int, np.ndarray] = {}
        for i in keys:
            found = self.fa.get_landmarks(frames[i])
            if found:
                known[i] = np.array(self._box_from_landmarks(found[0], h), dtype=np.float32)
        if not known:
            raise ValueError("no face found in the avatar video")
        idx = np.array(sorted(known))
        vals = np.stack([known[i] for i in idx])
        boxes = np.stack([np.interp(np.arange(n), idx, vals[:, k]) for k in range(4)], axis=1)
        return [tuple(int(round(v)) for v in b) for b in boxes]

    # ---------------- audio ----------------

    def _whisper_chunks(self, pcm16k: np.ndarray, fps: int):
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, pcm16k, 16000)
            feats, length = self.audio.get_audio_feature(tmp.name)
        return self.audio.get_whisper_chunk(
            feats, self.device, self.unet.model.dtype, self.whisper, length,
            fps=fps, audio_padding_length_left=AUDIO_PAD_LEFT, audio_padding_length_right=AUDIO_PAD_RIGHT,
        )

    # ---------------- main ----------------

    def __call__(self, frames_rgb: np.ndarray, pcm16k: np.ndarray, fps: int = 25) -> np.ndarray:
        """frames_rgb: [T, H, W, 3] uint8 avatar video; pcm16k: mono float32 speech. Returns re-synced RGB frames."""
        import cv2

        torch = self.torch
        with torch.no_grad():
            chunks = self._whisper_chunks(pcm16k, fps)
            n = min(len(frames_rgb), len(chunks))
            frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames_rgb[:n]]
            boxes = self._boxes(frames_rgb[:n])

            latents = []
            for (x1, y1, x2, y2), frame in zip(boxes, frames_bgr):
                crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
                latents.append(self.vae.get_latents_for_unet(crop))

            out = np.array(frames_rgb[:n], copy=True)
            for s in range(0, n, self.batch_size):
                e = min(n, s + self.batch_size)
                audio_feat = self.pe(chunks[s:e].to(self.device))
                lat = torch.cat(latents[s:e], dim=0).to(self.device, dtype=self.unet.model.dtype)
                pred = self.unet.model(lat, self.timesteps, encoder_hidden_states=audio_feat).sample
                for j, face in enumerate(self.vae.decode_latents(pred)):
                    i = s + j
                    x1, y1, x2, y2 = boxes[i]
                    face = cv2.resize(face.astype(np.uint8), (x2 - x1, y2 - y1))
                    merged = self._get_image(frames_bgr[i], face, [x1, y1, x2, y2], mode=PARSING_MODE, fp=self.parser)
                    out[i] = cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)
        return out
