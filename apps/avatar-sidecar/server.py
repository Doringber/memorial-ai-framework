"""Avatar video sidecar: reply audio in, lip-synced talking-head mp4 out.

Wraps avatarforcing-mlx (AvatarForcing, CVPR 2026, ported to Apple Silicon)
behind two local HTTP endpoints for the API:

    GET  /health   -> 200 once the model and portrait are loaded
    POST /render   body = audio bytes (mp3/wav/ogg...), header X-Max-Seconds
                   -> video/mp4 (512x512, 25 fps, with the audio track)

Optional passes, each off by default:
    --lipsync musetalk      MuseTalk 1.5 redraws the mouth from Whisper features
                            (multilingual, so Hebrew etc. sync better)
    --face-restore gfpgan   GFPGAN v1.4 sharpens the face (eyes, skin, glasses);
                            runs in its own venv via face_restore.py

Requests are queued and rendered one at a time on the main thread: MLX streams
are thread-local, and one render already uses the whole GPU. A long render
holds the connection until it finishes; the API's timeout must cover it.

Run it in its own environment (see README.md) — it pins transformers 4.30.2,
which the API's embedding/reranker stack can't use.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

log = logging.getLogger("avatar-sidecar")

MAX_BODY_BYTES = 25 * 1024 * 1024
RENDER_WAIT_S = 3600.0
HERE = Path(__file__).resolve().parent


@dataclass
class Job:
    audio: bytes
    max_seconds: float
    done: threading.Event = field(default_factory=threading.Event)
    video: bytes | None = None
    error: Exception | None = None


class Renderer:
    def __init__(
        self,
        image_path: str,
        *,
        crop: bool,
        weights: str | None,
        seed: int,
        a_cfg: float,
        lip_delay_frames: int,
        musetalk_home: str | None = None,
        gfpgan_python: str | None = None,
        gfpgan_weights: str | None = None,
    ) -> None:
        from avatarforcing_mlx import media
        from avatarforcing_mlx.pipeline import AvatarPipeline

        self.media = media
        self.seed = seed
        self.lip_delay_frames = lip_delay_frames
        self.gfpgan_python = gfpgan_python
        self.gfpgan_weights = Path(gfpgan_weights).expanduser() if gfpgan_weights else None

        if crop:
            from avatarforcing_mlx.preprocess import crop_face
            from PIL import Image

            image = crop_face(np.array(Image.open(image_path).convert("RGB")))
        else:
            image = media.load_image(image_path)

        started = time.perf_counter()
        self.pipe = AvatarPipeline(weights, a_cfg_scale=a_cfg)
        self.pipe.set_reference(image)
        log.info("Pipeline ready in %.1fs", time.perf_counter() - started)

        self.lipsync = None
        if musetalk_home:
            from lipsync import MuseTalkLipSync

            started = time.perf_counter()
            self.lipsync = MuseTalkLipSync(musetalk_home)
            log.info("MuseTalk lip sync ready in %.1fs", time.perf_counter() - started)

        if gfpgan_python:
            if not self.gfpgan_weights or not self.gfpgan_weights.is_file():
                raise FileNotFoundError("--gfpgan-weights must point at GFPGANv1.4.pth")
            log.info("GFPGAN face restoration enabled")

    def warm_up(self) -> None:
        """First render pays for compilation; do it before the first real request."""
        started = time.perf_counter()
        quiet = (np.random.default_rng(0).standard_normal(2 * self.media.SAMPLE_RATE) * 1e-3).astype(np.float32)
        self._frames(quiet)
        log.info("Warm-up render done in %.1fs", time.perf_counter() - started)

    def _frames(self, pcm: np.ndarray) -> np.ndarray:
        media = self.media
        n_frames = int(np.ceil(len(pcm) * media.FPS / media.SAMPLE_RATE))
        wa = self.pipe.audio_features(pcm, n_frames)
        k = self.lip_delay_frames
        if 0 < k < n_frames:
            # The model's mouth runs ahead of the sound on non-English speech (measured
            # ~120-160 ms on Hebrew TTS); hold the first audio frame to line them up.
            import mlx.core as mx

            wa = mx.concatenate([mx.repeat(wa[:, :1], k, axis=1), wa[:, :-k]], axis=1)
        blocks = [media.to_uint8(pix) for _, pix, _ in self.pipe.stream(wa, seed=self.seed)]
        return np.concatenate(blocks)[:n_frames]

    def _restore(self, frames_rgb: np.ndarray) -> np.ndarray:
        """Run GFPGAN in its own interpreter over a folder of PNG frames."""
        import cv2

        worker = HERE / "face_restore.py"
        with tempfile.TemporaryDirectory(prefix="gfpgan-") as tmp:
            src, dst = Path(tmp) / "in", Path(tmp) / "out"
            src.mkdir()
            for i, frame in enumerate(frames_rgb):
                cv2.imwrite(str(src / f"{i:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            # facexlib writes its detector weights into the working directory, so run it
            # from the weights folder rather than from the repo.
            subprocess.run(
                [self.gfpgan_python, str(worker), str(src), str(dst), "--weights", str(self.gfpgan_weights)],
                cwd=self.gfpgan_weights.parent,
                check=True,
            )
            out = []
            for i in range(len(frames_rgb)):
                img = cv2.imread(str(dst / f"{i:05d}.png"))
                if img is None:
                    raise RuntimeError(f"GFPGAN produced no frame {i}")
                out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return np.stack(out)

    def _render_frames(self, pcm: np.ndarray) -> np.ndarray:
        frames = self._frames(pcm)
        if self.lipsync is not None:
            frames = self.lipsync(frames, pcm)
        if self.gfpgan_python:
            restored = self._restore(frames)
            if self.lipsync is not None:
                # GFPGAN also redraws the lips and can ghost the mouth; keep MuseTalk's mouth.
                from mouth_blend import blend_mouth

                restored = blend_mouth(frames, restored, self.lipsync.fa)
            frames = restored
        return frames

    def render(self, audio: bytes, max_seconds: float) -> bytes:
        media = self.media
        with tempfile.TemporaryDirectory(prefix="avatar-") as tmp:
            src = Path(tmp) / "speech"
            src.write_bytes(audio)
            pcm = media.load_audio(str(src))[: int(max_seconds * media.SAMPLE_RATE)]
            if len(pcm) < media.SAMPLE_RATE // 10:
                raise ValueError("audio is shorter than 0.1 s")

            started = time.perf_counter()
            frames = self._render_frames(pcm)
            out = Path(tmp) / "out.mp4"
            media.write_video(str(out), frames, pcm)
            log.info(
                "Rendered %.1fs of video in %.1fs",
                len(frames) / media.FPS,
                time.perf_counter() - started,
            )
            return out.read_bytes()


def make_handler(jobs: "queue.Queue[Job]", default_max_seconds: float) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:  # route access logs through logging
            log.debug(fmt, *args)

        def _reply(self, code: int, body: bytes, content_type: str = "text/plain; charset=utf-8") -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._reply(200, b"ok")
            else:
                self._reply(404, b"not found")

        def do_POST(self) -> None:
            if self.path != "/render":
                self._reply(404, b"not found")
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                self._reply(413 if length > 0 else 400, b"audio body missing or too large")
                return
            try:
                max_seconds = min(float(self.headers.get("X-Max-Seconds") or default_max_seconds), 120.0)
            except ValueError:
                max_seconds = default_max_seconds
            job = Job(audio=self.rfile.read(length), max_seconds=max_seconds)
            jobs.put(job)
            if not job.done.wait(RENDER_WAIT_S):
                self._reply(504, b"render timed out")
                return
            if isinstance(job.error, ValueError):
                self._reply(422, str(job.error).encode())
            elif job.error is not None:
                self._reply(500, b"render failed")
            else:
                self._reply(200, job.video, "video/mp4")

    return Handler


def serve_jobs(renderer: Renderer, jobs: "queue.Queue[Job]") -> None:
    """Runs on the main thread: MLX streams are thread-local."""
    while True:
        job = jobs.get()
        try:
            job.video = renderer.render(job.audio, job.max_seconds)
        except Exception as e:  # noqa: BLE001 - reported to the waiting request
            if not isinstance(e, ValueError):
                log.exception("Render failed")
            job.error = e
        finally:
            job.done.set()


def main() -> None:
    ap = argparse.ArgumentParser(description="Local talking-head video sidecar (avatarforcing-mlx)")
    ap.add_argument("--image", required=True, help="portrait of the person (front-facing, one face)")
    ap.add_argument("--crop", action="store_true", help="detect and crop the face (needs the 'crop' extra)")
    ap.add_argument("--host", default="127.0.0.1", help="bind address; keep it on loopback")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--weights", default=None, help="model dir (default: avatarforcing-mlx's cache)")
    ap.add_argument("--seed", type=int, default=25)
    ap.add_argument("--max-seconds", type=float, default=60.0)
    ap.add_argument("--a-cfg", type=float, default=3.0,
                    help="audio guidance; higher = mouth follows speech more (upstream default 2.0)")
    ap.add_argument("--lip-delay-frames", type=int, default=3,
                    help="delay mouth motion by N frames @25fps to match the audio (0 for English)")
    ap.add_argument("--lipsync", choices=["none", "musetalk"], default="none",
                    help="second pass that re-syncs the mouth to the audio")
    ap.add_argument("--musetalk-home", default=os.environ.get("MUSETALK_HOME", "~/.cache/musetalk-src"),
                    help="MuseTalk clone with models/ (see README)")
    ap.add_argument("--face-restore", choices=["none", "gfpgan"], default="none",
                    help="face restoration pass (eyes, skin, glasses)")
    ap.add_argument("--gfpgan-python", default=os.environ.get("GFPGAN_PYTHON", "~/.cache/gfpgan-venv/bin/python"),
                    help="interpreter of the GFPGAN venv")
    ap.add_argument("--gfpgan-weights", default="~/.cache/gfpgan-weights/GFPGANv1.4.pth")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    use_musetalk = args.lipsync == "musetalk"
    use_gfpgan = args.face_restore == "gfpgan"
    renderer = Renderer(
        args.image,
        crop=args.crop,
        weights=args.weights,
        seed=args.seed,
        a_cfg=args.a_cfg,
        lip_delay_frames=args.lip_delay_frames,
        musetalk_home=str(Path(args.musetalk_home).expanduser()) if use_musetalk else None,
        gfpgan_python=str(Path(args.gfpgan_python).expanduser()) if use_gfpgan else None,
        gfpgan_weights=args.gfpgan_weights if use_gfpgan else None,
    )
    if not args.no_warmup:
        renderer.warm_up()

    jobs: "queue.Queue[Job]" = queue.Queue()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(jobs, args.max_seconds))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
    log.info("Avatar sidecar listening on http://%s:%d", args.host, args.port)
    try:
        serve_jobs(renderer, jobs)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
