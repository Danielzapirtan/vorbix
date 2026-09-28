# Copilot instructions for vorbix

## Project overview

This repository contains a self-contained Flask app that transcribes audio in two passes (Romanian and English), then merges the results into a single bilingual transcript. The app exposes a small web UI and JSON API in `app.py`, with the main behavior centered around backend selection, timed segment alignment, and English deduplication.

The service supports two backends:
- `whispermlx` for speaker diarization, requiring `HF_TOKEN`
- `faster-whisper` without diarization

The relevant logic is concentrated in `app.py`; tests are in `test_app.py`. `test.sh` is a manual end-to-end transcription script for local use.

## Setup and environment

Use the repo-local virtual environment and install dependencies from `requirements.txt`:

```bash
cd /home/daniel/code/Danielzapirtan/vorbix
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Important runtime variables:
- `HF_TOKEN`: required for `whispermlx` diarization
- `WHISPER_BACKEND`: default backend (`whispermlx` by default)
- `WHISPER_DEVICE`: default device (`cpu`)
- `WHISPER_MODEL`: model name (`large-v3`)
- `TRANSCRIPTIONS_DIR`: output directory for generated JSON files
- `DEDUP_SIM_THRESHOLD`: similarity threshold for English deduplication (`0.90`)

## Build, test, and validation commands

There is no dedicated lint task or formatter config in this repo. Validation is done through the Python unittest suite.

Run the full test file:

```bash
source .venv/bin/activate
python -m unittest test_app.py
```

Run a single test by name:

```bash
source .venv/bin/activate
python -m unittest test_app.FasterWhisperTests.test_unknown_backend_is_rejected
```

Run the app locally:

```bash
source .venv/bin/activate
python app.py
```

Then open `http://localhost:5002`.

For a full local transcription run using the shell script:

```bash
source .venv/bin/activate
export HF_TOKEN=...
source test.sh example.m4a
```

`test.sh` expects a local audio file path and uses the `whispermlx` flow to produce outputs under `$HOME/transcriptions/{ro,en}/` plus merged JSON output.

## High-level architecture

### `app.py`

This is the main application module. It handles:
- Flask app setup and routing
- backend selection (`whispermlx` vs `faster-whisper`)
- file upload and temp workdir lifecycle
- per-language transcription execution
- language scoring and overlap matching between Romanian and English segments
- deduplication of near-identical English segments before final output

Key functions to understand before editing behavior:
- `score(seg, expected_lang)`: language-likelihood heuristic for a segment
- `find_overlap(seg, candidates)`: chooses the best English match for a Romanian segment by time overlap
- `merge_transcriptions(ro_data, en_data, source_file)`: combines both passes into one merged result
- `dedupe_english_segments(merged, threshold=SIM_THRESHOLD)`: removes repeated English text that is effectively identical
- `run_whisper(audio_path, output_dir, language, backend)`: dispatches to the selected backend

### Output format

The app writes JSON outputs for each language pass and writes a merged JSON file with fields such as:
- `source_file`
- `segments`
- `turns`
- `full_text`
- `deduped_count`

The merged file is stored under `TRANSCRIPTIONS_DIR` and the route `/download/<path:filename>` serves it back.

### Test coverage

`test_app.py` verifies the core web behavior and backend selection:
- both backends are present in the UI
- invalid backend values are rejected with HTTP 400
- the selected backend is used for both Romanian and English passes
- `faster_whisper_transcribe()` normalizes output to the merged schema used elsewhere

## Key conventions in this repo

- The backend is not a free-form value; it is validated against `BACKENDS = ("whispermlx", "faster-whisper")` and rejected otherwise.
- The transcription flow deliberately runs twice: once for Romanian and once for English, then merges them using overlap and language-scoring heuristics rather than a single pass.
- Output directories are split by language (`RO_DIR` and `EN_DIR`) under the global transcription root, and the merge step writes a final `*.merged.json` file.
- The deduplication logic keeps the last near-identical English segment and removes previous duplicates when similarity exceeds `DEDUP_SIM_THRESHOLD`.
- The web route `/transcribe` expects a multipart upload with a file under the `audio` key; it stores the uploaded file in a temporary directory and removes it in a `finally` block.
- The service is intentionally environment-driven: model/device/backend defaults come from environment variables rather than code constants for local deployment flexibility.

## When making changes

- Prefer editing `app.py` and `test_app.py` together when changing transcription behavior or route expectations.
- Keep the backend contract stable: both routes and tests assume only `whispermlx` or `faster-whisper`.
- If changing output structure, update the merged result assertions in `test_app.py` together with the implementation.
- If you add or change environment variables, document them in the same style as the existing config variables near the top of `app.py`.
