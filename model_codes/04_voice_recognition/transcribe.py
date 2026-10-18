import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import whisper

from config import (
    MODEL_NAME,
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    PROCESSED_DIR,
    DEVICE,
)

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe Hindi or English audio using Whisper."
    )
    parser.add_argument("--audio", required=True, help="Path to the audio file")
    parser.add_argument(
        "--language",
        choices=SUPPORTED_LANGUAGES,
        default=DEFAULT_LANGUAGE,
        help="Language code: en or hi. Omit for automatic detection."
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size"
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Loading Whisper model: {args.model}")
    model = whisper.load_model(args.model, device=DEVICE)

    result = model.transcribe(
        str(audio_path),
        language=args.language,
        fp16=False
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    transcript = result.get("text", "").strip()
    output = {
        "audio_file": audio_path.name,
        "model": args.model,
        "language": result.get("language"),
        "transcript": transcript,
        "segments": [
            {
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text", "").strip()
            }
            for segment in result.get("segments", [])
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat()
    }

    txt_path = PROCESSED_DIR / f"{audio_path.stem}.txt"
    json_path = PROCESSED_DIR / f"{audio_path.stem}.json"

    txt_path.write_text(transcript, encoding="utf-8")
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\nTranscription:")
    print(transcript)
    print(f"\nText saved: {txt_path}")
    print(f"JSON saved: {json_path}")

if __name__ == "__main__":
    main()
