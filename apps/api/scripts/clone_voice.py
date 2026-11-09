"""Create an ElevenLabs Instant Voice Clone and write its id to $MEMORIAL_DATA_HOME/.env.

Uploads the recordings to ElevenLabs: requires ALLOW_EXTERNAL_SERVICES=true.
Only use recordings where you have confirmed the speaker is David.

Usage (from apps/api):
    .venv/bin/python -m scripts.clone_voice ~/MemorialData/data/voice_samples/roni_voice_1.wav ...
    .venv/bin/python -m scripts.clone_voice --list clips.txt --name David
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import MEMORIAL_DATA_HOME, settings  # noqa: E402

ENV_FILE = MEMORIAL_DATA_HOME / ".env"
_NEEDS_CONVERSION = {".opus", ".ogg", ".webm"}


def _to_mp3(path: Path, tmp_dir: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found on PATH (brew install ffmpeg)")
    out = tmp_dir / f"{path.stem}.mp3"
    subprocess.run([ffmpeg, "-y", "-i", str(path), "-q:a", "2", str(out)], capture_output=True, check=True)
    return out


def create_clone(audio_files: list[Path], name: str) -> str:
    from elevenlabs import ElevenLabs

    client = ElevenLabs(api_key=settings.elevenlabs_api_key)
    print(f"Creating voice clone '{name}' from {len(audio_files)} file(s)...")

    with tempfile.TemporaryDirectory() as tmp:
        prepared = [
            _to_mp3(f, Path(tmp)) if f.suffix.lower() in _NEEDS_CONVERSION else f for f in audio_files
        ]
        handles = [p.open("rb") for p in prepared]
        try:
            response = client.voices.ivc.create(
                name=name,
                files=handles,
                remove_background_noise=True,
                description="Memorial voice clone",
            )
        finally:
            for h in handles:
                h.close()
    return response.voice_id


def save_voice_id(voice_id: str) -> None:
    content = ENV_FILE.read_text(encoding="utf-8") if ENV_FILE.exists() else ""
    if "ELEVENLABS_VOICE_ID=" in content:
        content = re.sub(r"ELEVENLABS_VOICE_ID=.*", f"ELEVENLABS_VOICE_ID={voice_id}", content)
    else:
        content += f"\nELEVENLABS_VOICE_ID={voice_id}\n"
    ENV_FILE.write_text(content, encoding="utf-8")
    print(f"Saved ELEVENLABS_VOICE_ID to {ENV_FILE}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path, help="Audio files (wav, mp3, opus, ...)")
    parser.add_argument("--list", type=Path, help="Text file with one audio path per line")
    parser.add_argument("--name", default="David", help="Voice name on ElevenLabs")
    args = parser.parse_args()

    if not settings.allow_external_services:
        print("This uploads recordings to ElevenLabs; set ALLOW_EXTERNAL_SERVICES=true first.", file=sys.stderr)
        return 1
    if not settings.elevenlabs_api_key:
        print("ELEVENLABS_API_KEY is not set in .env", file=sys.stderr)
        return 1

    files: list[Path] = list(args.files)
    if args.list:
        files += [Path(line.strip()) for line in args.list.read_text().splitlines() if line.strip()]
    if not files:
        parser.print_help()
        return 2

    missing = [f for f in files if not f.exists()]
    if missing:
        for f in missing:
            print(f"File not found: {f}", file=sys.stderr)
        return 1

    save_voice_id(create_clone(files, args.name))
    print("Restart the API to use the new voice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
