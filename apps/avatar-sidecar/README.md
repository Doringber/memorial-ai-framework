# Avatar video sidecar (optional)

Turns each spoken reply into a short **talking-head video** of the person: one
portrait photo + the reply's synthesized voice → a lip-synced 512×512 mp4.
The Telegram bot sends it with the 🎥 button or in `/video` mode.

It wraps [avatarforcing-mlx](https://github.com/johnrobinsn/avatarforcing-mlx),
the Apple Silicon port of [AvatarForcing](https://github.com/TaekyungKi/AvatarForcing)
(Ki et al., CVPR 2026), and runs as its own local process because its model
stack pins `transformers==4.30.2`, which the API's embedding and reranker models
can't use.

```
reply text → TTS (mp3) → API ──POST /render──► sidecar (portrait loaded once)
                          ◄──── video/mp4 ────┘
          → Telegram sendVideo (captioned as AI-generated)
```

## Requirements

- Apple Silicon Mac, 32 GB+ unified memory recommended (~6 GB used while rendering).
- A working voice: `TTS_PROVIDER` must produce audio first. Video uses the same speech.
- [uv](https://docs.astral.sh/uv/). No ffmpeg binary needed (PyAV handles media).

Speed reported upstream on an M5 (32 GB): ~1.7–1.9× realtime, so a 15 s reply
renders in roughly 8–10 s, plus a one-time warm-up at startup. The optional
quality passes below are much slower.

## License — read this first

avatarforcing-mlx and the AvatarForcing weights are **CC BY-NC 4.0**; the motion
autoencoder weights (from FLOAT) are **CC BY-NC-ND 4.0**. **Non-commercial use
only.** Nothing from them is vendored in this repository (which is MIT): they
are installed from their own repo and the weights are downloaded from the
authors' sources onto your machine.

The optional passes have their own licenses: MuseTalk is MIT, GFPGAN is
Apache-2.0 (its face detector and parser from facexlib are also downloaded, not
vendored). None of their code or weights is stored in this repository.

## Setup

```bash
cd apps/avatar-sidecar
uv sync                                   # Python 3.12, MLX + PyTorch MPS, separate .venv
uv run avatarforcing-mlx download         # ~1.4 GB into ~/.cache/avatarforcing-mlx (one time, needs internet)
```

Put one clear, front-facing photo of the person (single face, good light,
neutral expression, mouth closed) in your data folder, outside the repo:

```bash
mkdir -p ~/MemorialData/data/avatar_video
cp /path/to/photo.jpg ~/MemorialData/data/avatar_video/portrait.jpg
chmod 600 ~/MemorialData/data/avatar_video/portrait.jpg
```

## Quality passes (optional, slower)

The base model alone is fast, but its Hebrew lip sync is weak and its face is
soft. Two extra passes address that; each is off by default:

| Flag | What it does | Cost per 27 s reply (M-series Mac) |
|---|---|---|
| `--lipsync musetalk` | MuseTalk 1.5 redraws the mouth from Whisper features (multilingual) | ~4.5 min |
| `--face-restore gfpgan` | GFPGAN v1.4 sharpens eyes, skin and glasses | ~8 min |

With both on, a 27 s reply takes about 10 minutes. The mouth from MuseTalk is
pasted back over GFPGAN's, because GFPGAN can ghost the lips on some frames.

**MuseTalk** (MIT): clone the repo and fetch its weights into `models/`
(`musetalkV15/`, `whisper/`, `sd-vae/`), following MuseTalk's README. The VAE must
be the safetensors file from `stabilityai/sd-vae-ft-mse`
(`diffusion_pytorch_model.safetensors`, with its `config.json`). The sidecar does
not load pickled weights.

```bash
git clone https://github.com/TMElyralab/MuseTalk ~/.cache/musetalk-src
# then fetch models/ as described in MuseTalk's README (MUSETALK_HOME can point elsewhere)
```

**GFPGAN** runs in its own venv, because it needs torch + basicsr:

```bash
uv venv --python 3.12 ~/.cache/gfpgan-venv
uv pip install --python ~/.cache/gfpgan-venv/bin/python gfpgan opencv-python-headless
# basicsr imports a module that torchvision 0.17+ removed; point it at the new name:
grep -rl functional_tensor ~/.cache/gfpgan-venv/lib/python3.12/site-packages/basicsr \
  | xargs sed -i '' 's/torchvision\.transforms\.functional_tensor/torchvision.transforms.functional/g'
mkdir -p ~/.cache/gfpgan-weights
curl -fL -o ~/.cache/gfpgan-weights/GFPGANv1.4.pth \
  https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
```

facexlib downloads its face detector on first use, into `~/.cache/gfpgan-weights`.

## Run

```bash
# base model only (fast, weaker Hebrew lips)
uv run python server.py --image ~/MemorialData/data/avatar_video/portrait.jpg --crop

# full quality
uv run python server.py --image ~/MemorialData/data/avatar_video/portrait.jpg --crop \
    --lipsync musetalk --face-restore gfpgan
# Avatar sidecar listening on http://127.0.0.1:8090
```

Then in `~/MemorialData/.env`:

```bash
AVATAR_VIDEO_ENABLED=true
AVATAR_SERVICE_URL=http://127.0.0.1:8090          # Docker: http://host.docker.internal:8090
```

Try it without the API:

```bash
curl -s --data-binary @reply.mp3 -H 'Content-Type: application/octet-stream' \
     -H 'X-Max-Seconds: 30' http://127.0.0.1:8090/render -o reply.mp4
```

No private audio is needed for a first test: `examples/make_dummy_speech.sh`
makes a short Hebrew and an English clip with macOS voices. Use any consenting
photo of a person as the portrait; this repository ships none.

## Notes

- **Local only.** The sidecar binds to `127.0.0.1`. The API refuses to start if
  `AVATAR_SERVICE_URL` points anywhere else (reply audio would leave the machine)
  unless `ALLOW_EXTERNAL_SERVICES=true`.
- **Language.** Without `--lipsync musetalk`, lip motion is driven by wav2vec2
  features (English-trained), so Hebrew lips sync loosely. MuseTalk's Whisper
  features are multilingual and are the fix for that. Check mouth shapes on a few
  real replies either way.
- **One at a time.** Requests are queued and rendered sequentially on the main
  thread (MLX requires it). A request waits in line. The API times out after
  `AVATAR_VIDEO_TIMEOUT_S` (1200 s by default, enough for a 30 s reply with both
  passes) and keeps the text reply.
- **Disclosure.** Every video is sent with a caption saying it is AI-generated.
  Keep it: this is a real person's face.
