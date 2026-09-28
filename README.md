# Bilingual transcription app

This self-contained Flask app transcribes audio in two passes (Romanian and
English) and merges the results using the existing language-scoring and
deduplication logic. Choose `whispermlx` or `faster-whisper` in the web form.
Speaker diarization is available with `whispermlx`; Faster Whisper results use
`null` for the speaker.

## Run on Linux

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open `http://localhost:5002`, upload an audio file, and select **Faster
Whisper**. The model is selected with `WHISPER_MODEL` (default: `large-v3`),
and Faster Whisper uses `WHISPER_DEVICE` (default: `cpu`; use `cuda` for a
supported NVIDIA setup). A live progress bar is shown while Faster Whisper
transcribes the Romanian and English passes. Set
`WHISPER_BACKEND=faster-whisper` to make it the default selection in the form.

`HF_TOKEN` is required for whispermlx's speaker diarization. It is not required
for Faster Whisper with public models.
