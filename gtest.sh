#!/usr/bin/env bash

AUDIO_FILE="$1"
GEMINI_API_KEY="${GEMINI_API_KEY:-}"
OUTPUT_DIR="$HOME/transcriptions"
VER=3.12

if [ -z "$GEMINI_API_KEY" ]; then
    echo "Please set GEMINI_API_KEY environment variable"
    exit 1
fi

# Ensure python and venv
if ! command -v python$VER &>/dev/null; then
    brew install python@$VER
fi
mkdir -p "$OUTPUT_DIR"
python$VER -m venv venv
source venv/bin/activate
python$VER -m pip install --upgrade pip
pip install google-generativeai

# Transcribe with Gemini (native code-switching support)
python$VER << EOF
import google.generativeai as genai
import os, json

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-3.5-transcribe")

with open("$AUDIO_FILE", "rb") as f:
    audio_data = f.read()

# Request transcription with language detection
response = model.generate_content([
    {"mime_type": "audio/mp4", "data": audio_data},
    "Transcribe this audio. The speaker mixes Romanian and English. "
    "For each segment, output: [LANG:xx] text. Use 'ro' for Romanian, 'en' for English."
])

# Parse response into your desired format
segments = []
for line in response.text.strip().split('\n'):
    if line.startswith('[LANG:'):
        lang = line[6:8]
        text = line[9:].strip()
        segments.append({"lang": lang, "text": text})

output = {"segments": segments, "full_text": response.text}
outfile = os.path.join("$OUTPUT_DIR", os.path.basename("$AUDIO_FILE").replace('.m4a', '.json'))
with open(outfile, 'w') as f:
    json.dump(output, f, indent=2)
print(f"Saved to {outfile}")
EOF
