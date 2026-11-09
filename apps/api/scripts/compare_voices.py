"""Compare voice cloning backends and settings side by side against the real reference clip.

Writes a self-contained HTML listening sheet to $MEMORIAL_DATA_HOME/eval/voice_ab/index.html
(outside Git) so family members can listen and judge naturalness, Hebrew pronunciation,
and closeness to David's real 15.2-second reference recording without fabricated
similarity scores.

Modes:
  --sync-elevenlabs-settings  Update the saved default settings on ElevenLabs voice
                              ELEVENLABS_VOICE_ID (stability=1.0, similarity_boost=0.95,
                              use_speaker_boost=True). Costs 0 credits.
  --local-only                Synthesize the test sentences using only the free local
                              Chatterbox Multilingual clone (0 credits, no external calls).
  --include-elevenlabs        Also synthesize the test sentences with ElevenLabs
                              (before vs after tuning) using the monthly character budget.

Usage (from apps/api):
    .venv/bin/python -m scripts.compare_voices --sync-elevenlabs-settings
    .venv/bin/python -m scripts.compare_voices --local-only
    .venv/bin/python -m scripts.compare_voices --include-elevenlabs
"""

from __future__ import annotations

import argparse
import html
import shutil
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from app.services import tts  # noqa: E402

DEFAULT_REFERENCE = Path.home() / "Downloads" / "reference-voice.wav"
STUDIO_EXAMPLES = (
    (
        "ElevenLabs Studio דוגמה 1 (ivc_s70_sim99_v4 — 15.3s)",
        Path.home() / "Downloads" / "ElevenLabs_2026-11-07T00_54_34_sample_voice_ivc_s70_sim99_v4.mp3",
        "studio_example_1.mp3",
    ),
    (
        "ElevenLabs Studio דוגמה 2 (ivc_s70_sim99_v4 — 4.0s)",
        Path.home() / "Downloads" / "ElevenLabs_2026-11-07T00_56_33_sample_voice_ivc_s70_sim99_v4.mp3",
        "studio_example_2.mp3",
    ),
)
OUT_DIR = settings.data_dir.parent / "eval" / "voice_ab"

TEST_SENTENCES = (
    ("greeting", "שלום לך יקר, איזה כיף לשמוע ממך היום."),
    ("everyday", "כלום מיוחד, סדר יום רגיל בבית. מה אצלכם?"),
    ("car", "הרכב תמיד נאמן, אבל צריך תחזוקה מסודרת ובדיקה של החשמל."),
)


def sync_elevenlabs_voice_settings() -> dict:
    """Update the saved default settings on the ElevenLabs voice object (0 credits)."""
    if not (settings.elevenlabs_api_key and settings.elevenlabs_voice_id):
        raise RuntimeError("ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID must be set")
    url = f"https://api.elevenlabs.io/v1/voices/{settings.elevenlabs_voice_id}/settings/edit"
    payload = {
        "stability": settings.elevenlabs_stability,
        "similarity_boost": settings.elevenlabs_similarity_boost,
        "style": 0.0,
        "use_speaker_boost": settings.elevenlabs_speaker_boost,
    }
    response = httpx.post(
        url,
        headers={"xi-api-key": settings.elevenlabs_api_key},
        json=payload,
        timeout=20,
    )
    response.raise_for_status()
    return payload


def _synthesize_elevenlabs_variant(
    text: str,
    out_path: Path,
    *,
    stability: float,
    similarity_boost: float,
    speaker_boost: bool,
) -> float:
    from elevenlabs import ElevenLabs, VoiceSettings

    spoken = tts.prepare_text_for_speech(text)
    if tts.elevenlabs_chars_used() + len(spoken) > settings.elevenlabs_monthly_char_limit:
        raise tts.VoiceUnavailableError("budget", "Monthly ElevenLabs character limit reached")
    tts._record_elevenlabs_chars(len(spoken))

    started = time.perf_counter()
    client = ElevenLabs(api_key=settings.elevenlabs_api_key, timeout=60)
    stream = client.text_to_speech.convert(
        voice_id=settings.elevenlabs_voice_id,
        text=spoken,
        model_id=settings.elevenlabs_model_id,
        language_code="he",
        output_format="mp3_44100_128",
        voice_settings=VoiceSettings(
            stability=stability,
            similarity_boost=similarity_boost,
            use_speaker_boost=speaker_boost,
        ),
        request_options={"max_retries": 0},
    )
    with out_path.open("wb") as f:
        for chunk in stream:
            f.write(chunk)
    return round(time.perf_counter() - started, 2)


def _write_html(reference_name: str, studio_refs: list[tuple[str, str]], rows: list[dict]) -> Path:
    studio_html = "\n".join(
        f"""
        <div class="sample">
          <div class="label">{html.escape(label)}</div>
          <audio controls preload="metadata" src="{html.escape(fname)}"></audio>
        </div>"""
        for label, fname in studio_refs
    )
    cards: list[str] = []
    for row in rows:
        players = "\n".join(
            f"""
            <div class="sample">
              <div class="label">{html.escape(item['label'])} <small>({item['seconds']}s)</small></div>
              <audio controls preload="metadata" src="{html.escape(item['file'])}"></audio>
            </div>"""
            for item in row["variants"]
        )
        cards.append(
            f"""
          <section class="card">
            <h2>{html.escape(row['id'])}</h2>
            <p class="sentence">{html.escape(row['text'])}</p>
            <div class="grid">{players}</div>
          </section>"""
        )

    page = f"""<!doctype html>
<html lang="he" dir="rtl">
<head>
  <meta charset="utf-8" />
  <title>השוואת קול — דוד (סימולציה)</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background: #fff8e6; color: #182127; margin: 0; padding: 32px; }}
    .wrap {{ max-width: 820px; margin: 0 auto; }}
    .notice {{ border-inline-start: 3px solid #2055e8; background: #fffdf5; padding: 10px 14px; margin-bottom: 24px; font-size: 14px; }}
    .card {{ background: #fffdf5; border: 1px solid #182127; border-radius: 6px; box-shadow: -5px 5px 0 #ffd43b; padding: 20px; margin-bottom: 20px; }}
    .sentence {{ font-size: 18px; font-weight: 600; margin: 6px 0 14px; }}
    .grid {{ display: grid; gap: 12px; }}
    .sample {{ border-top: 1px dashed #c8c5b6; padding-top: 10px; }}
    .label {{ font-size: 14px; margin-bottom: 6px; font-weight: 500; }}
    .label small {{ color: #525b60; font-weight: 400; }}
    audio {{ width: 100%; height: 38px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>השוואת דגימות קול מול הקלטת המקור</h1>
    <div class="notice">
      האזנה עיוורת לבני המשפחה בלבד. אין כאן ציוני דמיון אוטומטיים — בדקו הגייה בעברית, טבעיות ודמיון להקלטה האמיתית.
    </div>
    <section class="card">
      <h2>הקלטת מקור אמיתית (15.2 שניות) + דוגמאות הייחוס מ-ElevenLabs (ivc_s70_sim99_v4)</h2>
      <div class="label">הקלטת מקור של דוד (15.2s)</div>
      <audio controls preload="metadata" src="{html.escape(reference_name)}"></audio>
      <div class="grid">{studio_html}</div>
    </section>
    {''.join(cards)}
  </div>
</body>
</html>
"""
    index_path = OUT_DIR / "index.html"
    index_path.write_text(page, encoding="utf-8")
    index_path.chmod(0o600)
    return index_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--sync-elevenlabs-settings", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--include-elevenlabs", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.chmod(0o700)

    if args.sync_elevenlabs_settings:
        updated = sync_elevenlabs_voice_settings()
        print("Synced ElevenLabs voice settings (0 credits):", updated)

    ref_copy = OUT_DIR / "reference.wav"
    if args.reference.is_file():
        shutil.copy2(args.reference, ref_copy)
        ref_copy.chmod(0o600)
        settings.local_voice_reference_wav = str(args.reference)

    studio_refs: list[tuple[str, str]] = []
    for label, src_path, dest_name in STUDIO_EXAMPLES:
        if src_path.is_file():
            dest = OUT_DIR / dest_name
            shutil.copy2(src_path, dest)
            dest.chmod(0o600)
            studio_refs.append((label, dest_name))

    rows: list[dict] = []
    for slug, sentence in TEST_SENTENCES:
        variants: list[dict] = []
        if tts._local_clone_available():
            local_out = OUT_DIR / f"{slug}-local.mp3"
            t0 = time.perf_counter()
            tts._synthesize_local_clone(tts.prepare_text_for_speech(sentence), str(local_out))
            local_out.chmod(0o600)
            variants.append(
                {
                    "label": "שיבוט מקומי חינמי (Chatterbox Multilingual)",
                    "file": local_out.name,
                    "seconds": round(time.perf_counter() - t0, 2),
                }
            )
        elif args.local_only:
            print(
                "Local voice clone model is not installed yet (chatterbox-tts). "
                "Install it in apps/api/.venv after approval to synthesize locally for free.",
                file=sys.stderr,
            )
            return 1

        if args.include_elevenlabs and not args.local_only:
            if not settings.tts_allowed:
                raise RuntimeError("Set ALLOW_ELEVENLABS_TTS=true to synthesize ElevenLabs comparison clips")
            prev_out = OUT_DIR / f"{slug}-elevenlabs-prev.mp3"
            sec_prev = _synthesize_elevenlabs_variant(
                sentence,
                prev_out,
                stability=0.5,
                similarity_boost=0.75,
                speaker_boost=False,
            )
            prev_out.chmod(0o600)
            variants.append(
                {
                    "label": "ElevenLabs (לפני כיול: stability=0.5, similarity=0.75)",
                    "file": prev_out.name,
                    "seconds": sec_prev,
                }
            )

            tuned_out = OUT_DIR / f"{slug}-elevenlabs-tuned.mp3"
            sec_tuned = _synthesize_elevenlabs_variant(
                sentence,
                tuned_out,
                stability=0.70,
                similarity_boost=0.99,
                speaker_boost=True,
            )
            tuned_out.chmod(0o600)
            variants.append(
                {
                    "label": "ElevenLabs v4 (אחרי כיול: stability=0.70, similarity=0.99, speaker_boost)",
                    "file": tuned_out.name,
                    "seconds": sec_tuned,
                }
            )

        if variants:
            rows.append({"id": slug, "text": sentence, "variants": variants})

    index_path = _write_html(ref_copy.name if ref_copy.exists() else "", studio_refs, rows)
    print(f"Wrote comparison sheet: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
