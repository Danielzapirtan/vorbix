import os
import re
import json
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from functools import lru_cache
from difflib import SequenceMatcher
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template_string

app = Flask(__name__)

# ---- Configuration ----
DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
HF_TOKEN = os.environ.get("HF_TOKEN")
TRANSCRIPTIONS_DIR = Path(os.environ.get("TRANSCRIPTIONS_DIR", Path.home() / "transcriptions"))
MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
DEFAULT_BACKEND = os.environ.get("WHISPER_BACKEND", "whispermlx").strip().lower()
BACKENDS = ("whispermlx", "faster-whisper")
if DEFAULT_BACKEND not in BACKENDS:
    raise ValueError(f"WHISPER_BACKEND must be one of: {', '.join(BACKENDS)}")
IVRIT = "ivrit-ai/pyannote-speaker-diarization-3.1"
ALTPYA = "pyannote/speaker-diarization-2.1"
PYA = "pyannote/speaker-diarization-community-1"

# Similarity threshold for considering two English segments "essentially the same sentence"
SIM_THRESHOLD = float(os.environ.get("DEDUP_SIM_THRESHOLD", "0.90"))

RO_DIR = TRANSCRIPTIONS_DIR / "ro"
EN_DIR = TRANSCRIPTIONS_DIR / "en"
RO_DIR.mkdir(parents=True, exist_ok=True)
EN_DIR.mkdir(parents=True, exist_ok=True)

_PROGRESS_LOCK = threading.Lock()
_TRANSCRIPTION_PROGRESS = {}
_PROGRESS_TTL_SECONDS = 3600


def _set_transcription_progress(progress_id, percentage, message, state="working"):
    if not progress_id:
        return
    now = time.monotonic()
    with _PROGRESS_LOCK:
        expired = [
            key for key, value in _TRANSCRIPTION_PROGRESS.items()
            if now - value["updated_at"] > _PROGRESS_TTL_SECONDS
        ]
        for key in expired:
            del _TRANSCRIPTION_PROGRESS[key]
        _TRANSCRIPTION_PROGRESS[progress_id] = {
            "percentage": round(max(0, min(100, percentage))),
            "message": message,
            "state": state,
            "updated_at": now,
        }


def _get_transcription_progress(progress_id):
    now = time.monotonic()
    with _PROGRESS_LOCK:
        progress = _TRANSCRIPTION_PROGRESS.get(progress_id)
        if progress and now - progress["updated_at"] > _PROGRESS_TTL_SECONDS:
            del _TRANSCRIPTION_PROGRESS[progress_id]
            progress = None
        if progress:
            return {
                "percentage": progress["percentage"],
                "message": progress["message"],
                "state": progress["state"],
            }
    return None


# ---- Merging logic (ported from the heredoc Python block) ----
RO_DIACRITICS = re.compile(r"[ăâîșțĂÂÎȘȚ]")
RO_WORDS = re.compile(
    r"\b(și|sau|dar|este|sunt|care|pentru|acest|această|foarte|"
    r"mulțumesc|bună|da|nu|ce|cum|unde|când| pentru că|"
    r"avem|aveți|trebuie|poate|face|făcut)\b",
    re.IGNORECASE,
)
EN_WORDS = re.compile(
    r"\b(the|and|or|but|is|are|which|for|this|that|very|"
    r"thanks|hello|yes|no|what|how|where|when|because|"
    r"have|need|can|make|made|would|could|should)\b",
    re.IGNORECASE,
)


def get_segments(data):
    if isinstance(data, dict) and "segments" in data:
        return data["segments"]
    if isinstance(data, dict) and "transcription" in data:
        return data["transcription"]
    if isinstance(data, list):
        return data
    return []


def score(seg, expected_lang):
    text = (seg.get("text") or "").strip()
    if not text:
        return -1e9

    avg_logprob = seg.get("avg_logprob", -1.0)
    no_speech = seg.get("no_speech_prob", 0.0)
    comp_ratio = seg.get("compression_ratio", 1.0)

    s = avg_logprob * 1.0
    s -= no_speech * 2.0
    s -= max(0.0, comp_ratio - 2.4) * 0.5

    ro_hits = len(RO_DIACRITICS.findall(text)) + len(RO_WORDS.findall(text))
    en_hits = len(EN_WORDS.findall(text))

    if expected_lang == "ro":
        s += ro_hits * 0.3
        s -= en_hits * 0.15
    else:
        s += en_hits * 0.3
        s -= ro_hits * 0.15

    if len(text.split()) <= 2:
        s *= 0.5

    return s


def find_overlap(seg, candidates, tol=0.5):
    best = None
    best_score = -1
    for c in candidates:
        ov_start = max(seg.get("start", 0), c.get("start", 0))
        ov_end = min(seg.get("end", 0), c.get("end", 0))
        overlap = max(0.0, ov_end - ov_start)
        if overlap > best_score:
            best_score = overlap
            best = c
    if best is None and candidates:
        best = min(candidates, key=lambda c: abs(c.get("start", 0) - seg.get("start", 0)))
        if abs(best.get("start", 0) - seg.get("start", 0)) > tol:
            best = None
    return best


def merge_adjacent(segs):
    out = []
    for s in segs:
        if out and out[-1]["lang"] == s["lang"] and out[-1].get("speaker") == s.get("speaker"):
            out[-1]["text"] += " " + s["text"]
            out[-1]["end"] = s["end"]
        else:
            out.append(dict(s))
    return out


def _normalize_text(t):
    """Lowercase, strip punctuation, collapse whitespace for comparison."""
    t = (t or "").lower()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _similarity(a, b):
    """Return similarity ratio (0..1) between two normalized strings."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def dedupe_english_segments(merged, threshold=SIM_THRESHOLD):
    """
    If 90%+ of two or more English segments (consecutive or not) are
    essentially the same sentence, discard all occurrences except the LAST one.

    Returns (new_merged, num_dropped).
    """
    en_indices = [i for i, s in enumerate(merged) if s.get("lang") == "en"]
    if len(en_indices) < 2:
        return merged, 0

    norm = {i: _normalize_text(merged[i].get("text")) for i in en_indices}

    # Union-Find
    parent = {i: i for i in en_indices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a_pos in range(len(en_indices)):
        ia = en_indices[a_pos]
        na = norm[ia]
        if not na:
            continue
        for b_pos in range(a_pos + 1, len(en_indices)):
            ib = en_indices[b_pos]
            nb = norm[ib]
            if not nb:
                continue
            la, lb = len(na), len(nb)
            if min(la, lb) / max(la, lb) < threshold:
                continue
            if _similarity(na, nb) >= threshold:
                union(ia, ib)

    groups = {}
    for i in en_indices:
        r = find(i)
        groups.setdefault(r, []).append(i)

    drop_indices = set()
    for _r, members in groups.items():
        if len(members) > 1:
            members_sorted = sorted(members, key=lambda i: (merged[i].get("start") or 0))
            # Keep only the last occurrence
            for i in members_sorted[:-1]:
                drop_indices.add(i)

    if not drop_indices:
        return merged, 0

    new_merged = [s for i, s in enumerate(merged) if i not in drop_indices]
    return new_merged, len(drop_indices)


def merge_transcriptions(ro_data, en_data, source_file):
    ro_segs = get_segments(ro_data)
    en_segs = get_segments(en_data)

    merged = []
    used_en_ids = set()

    for ro_seg in ro_segs:
        en_match = find_overlap(ro_seg, en_segs)
        if en_match is not None:
            used_en_ids.add(id(en_match))

        ro_s = score(ro_seg, "ro")
        en_s = score(en_match, "en") if en_match else -1e9

        if en_match and en_s > ro_s:
            chosen = en_match
            lang = "en"
        else:
            chosen = ro_seg
            lang = "ro"

        merged.append({
            "start": chosen.get("start"),
            "end": chosen.get("end"),
            "lang": lang,
            "speaker": chosen.get("speaker"),
            "text": (chosen.get("text") or "").strip(),
            "ro_text": (ro_seg.get("text") or "").strip(),
            "en_text": (en_match.get("text") or "").strip() if en_match else None,
            "scores": {"ro": ro_s, "en": en_s},
        })

    for en_seg in en_segs:
        if id(en_seg) in used_en_ids:
            continue
        merged.append({
            "start": en_seg.get("start"),
            "end": en_seg.get("end"),
            "lang": "en",
            "speaker": en_seg.get("speaker"),
            "text": (en_seg.get("text") or "").strip(),
            "ro_text": None,
            "en_text": (en_seg.get("text") or "").strip(),
            "scores": {"ro": None, "en": score(en_seg, "en")},
        })

    merged.sort(key=lambda s: (s.get("start") or 0))

    # --- Deduplicate near-identical English segments ---
    merged, dropped = dedupe_english_segments(merged)
    if dropped:
        print(f"Deduplicated {dropped} near-identical English segment(s).")

    return {
        "source_file": source_file,
        "segments": merged,
        "turns": merge_adjacent(merged),
        "full_text": " ".join(s["text"] for s in merged if s["text"]),
        "deduped_count": dropped,
    }


# ---- Whisper invocation ----
@lru_cache(maxsize=2)
def get_faster_whisper_model(model_name, device):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Faster Whisper is not installed. Install the faster-whisper dependency."
        ) from exc
    return WhisperModel(model_name, device=device, compute_type="default")


def faster_whisper_transcribe(audio_path, output_dir, language, progress_callback=None):
    model = get_faster_whisper_model(MODEL, DEVICE)
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    output_segments = []
    duration = getattr(info, "duration", 0)
    for segment in segments:
        output_segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "avg_logprob": segment.avg_logprob,
            "no_speech_prob": segment.no_speech_prob,
            "compression_ratio": segment.compression_ratio,
            "speaker": None,
        })
        if progress_callback and duration and duration > 0:
            progress_callback(segment.end / duration)
    if progress_callback:
        progress_callback(1)

    data = {
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": output_segments,
    }
    output_file = output_dir / f"{Path(audio_path).stem}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data


def run_whisper(audio_path, output_dir, language, backend, progress_callback=None):
    if backend == "faster-whisper":
        return faster_whisper_transcribe(
            audio_path, output_dir, language, progress_callback=progress_callback
        )

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN environment variable is not set")

    cmd = [
        "whispermlx", str(audio_path),
        "--compression_ratio_threshold", "2.4",
        "--condition_on_previous_text", "False",
        "--device", DEVICE,
        "--diarize",
        "--hf_token", HF_TOKEN,
        "--language", language,
        "--logprob_threshold", "-1.0",
        "--model", MODEL,
        "--no_speech_threshold", "0.6",
        "--output_dir", str(output_dir),
        "--output_format", "json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"whispermlx ({language}) failed (rc={proc.returncode}):\n{proc.stderr}"
        )
    return proc.stdout


# ---- Flask routes ----
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Transcription Service</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #172334;
      background: #f3f6fb;
      font-synthesis: none;
      text-rendering: optimizeLegibility;
      -webkit-font-smoothing: antialiased;
      --muted: #68778b;
      --line: #e1e8f1;
      --blue: #315be8;
    }
    * { box-sizing: border-box; }
    body {
      min-height: 100vh;
      margin: 0;
      padding: 52px 20px 72px;
      background:
        radial-gradient(ellipse at 50% -10%, rgba(94, 132, 255, .17), transparent 42%),
        #f3f6fb;
    }
    main { width: min(100%, 760px); margin: 0 auto; }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 0 0 30px 2px;
      color: #53647b;
      font-size: 13px;
      font-weight: 700;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .brand-mark {
      display: grid;
      width: 34px;
      height: 34px;
      place-items: center;
      border-radius: 11px;
      background: #315be8;
      color: white;
      font-size: 16px;
      letter-spacing: 0;
      box-shadow: 0 5px 14px rgba(49, 91, 232, .24);
    }
    .card {
      overflow: hidden;
      border: 1px solid rgba(221, 229, 240, .9);
      border-radius: 22px;
      background: #fff;
      box-shadow: 0 18px 55px rgba(29, 48, 82, .09);
    }
    .intro { padding: 42px 46px 32px; }
    .eyebrow {
      margin: 0 0 12px;
      color: var(--blue);
      font-size: 12px;
      font-weight: 750;
      letter-spacing: .11em;
      text-transform: uppercase;
    }
    h1 {
      margin: 0;
      color: #142238;
      font-size: clamp(30px, 5vw, 42px);
      letter-spacing: -.045em;
      line-height: 1.12;
    }
    .description {
      max-width: 570px;
      margin: 15px 0 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.7;
    }
    form {
      display: grid;
      gap: 24px;
      padding: 30px 46px 38px;
      border-top: 1px solid #edf1f6;
      background: #fcfdff;
    }
    .field { display: grid; gap: 9px; }
    label, .field-label {
      color: #26364b;
      font-size: 13px;
      font-weight: 700;
    }
    .hint { color: var(--muted); font-size: 12px; line-height: 1.5; }
    input[type=file], select {
      width: 100%;
      min-height: 48px;
      border: 1px solid #d7e0ec;
      border-radius: 10px;
      background: #fff;
      color: #25364d;
      font: inherit;
    }
    input[type=file] { padding: 7px; color: var(--muted); font-size: 13px; }
    input[type=file]::file-selector-button {
      margin-right: 12px;
      padding: 9px 13px;
      border: 0;
      border-radius: 7px;
      background: #edf2ff;
      color: #315be8;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    select { padding: 0 13px; }
    input:focus-visible, select:focus-visible, button:focus-visible {
      outline: 3px solid rgba(49, 91, 232, .25);
      outline-offset: 2px;
    }
    button {
      display: inline-flex;
      min-height: 49px;
      align-items: center;
      justify-content: center;
      gap: 9px;
      padding: 0 20px;
      border: 0;
      border-radius: 10px;
      background: var(--blue);
      color: #fff;
      font: inherit;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 6px 14px rgba(49, 91, 232, .2);
      transition: background .15s ease, transform .15s ease, box-shadow .15s ease;
    }
    button:hover { transform: translateY(-1px); background: #244bd0; box-shadow: 0 9px 18px rgba(49, 91, 232, .25); }
    button:disabled { cursor: wait; opacity: .72; transform: none; }
    .button-icon { font-size: 17px; line-height: 1; }
    .status {
      display: none;
      margin: 20px 0 0;
      padding: 14px 17px;
      border: 1px solid var(--line);
      border-radius: 11px;
      background: #fff;
      color: #46566c;
      font-size: 13px;
      line-height: 1.55;
      overflow-wrap: anywhere;
    }
    .status.visible { display: block; }
    .status[data-state="error"] { border-color: #f3d0d0; background: #fff8f8; color: #a13232; }
    .status[data-state="success"] { border-color: #ccebd9; background: #f5fcf7; color: #236741; }
    .progress[hidden] { display: none; }
    .progress {
      display: grid;
      gap: 8px;
      margin-top: 10px;
      padding: 13px 17px;
      border: 1px solid var(--line);
      border-radius: 11px;
      background: #fff;
    }
    .progress-label { color: #46566c; font-size: 12px; }
    .progress-track {
      height: 9px;
      overflow: hidden;
      border-radius: 999px;
      background: #e9eef7;
    }
    .progress-value {
      width: 0;
      height: 100%;
      border-radius: inherit;
      background: var(--blue);
      transition: width .25s ease;
    }
    .result-wrap {
      margin-top: 18px;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
      box-shadow: 0 8px 24px rgba(29, 48, 82, .05);
    }
    .result-heading {
      margin: 0;
      padding: 15px 19px;
      border-bottom: 1px solid #edf1f6;
      color: #26364b;
      font-size: 13px;
      font-weight: 700;
    }
    pre {
      margin: 0;
      padding: 20px;
      overflow: auto;
      max-height: 60vh;
      color: #31445f;
      background: #fbfcfe;
      font: 12px/1.65 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .footnote { margin: 17px 3px 0; color: #8491a3; font-size: 12px; text-align: center; }
    @media (max-width: 560px) {
      body { padding: 25px 14px 40px; }
      .brand { margin-bottom: 19px; }
      .intro { padding: 30px 24px 24px; }
      form { gap: 21px; padding: 24px; }
      button { width: 100%; }
    }
  </style>
</head>
<body>
  <main>
    <div class="brand"><span class="brand-mark" aria-hidden="true">V</span> Vorbix transcription</div>
    <section class="card" aria-labelledby="page-title">
      <div class="intro">
        <p class="eyebrow">Audio to text</p>
        <h1 id="page-title">Bilingual transcription,<br>made simple.</h1>
        <p class="description">Get a single Romanian and English transcript from your audio. Vorbix compares both language passes and combines the clearest segments.</p>
      </div>
      <form id="f">
        <div class="field">
          <label for="audio">Choose an audio file</label>
          <input type="file" id="audio" name="audio" accept="audio/*,.m4a" required>
          <span class="hint">Select an audio recording to transcribe.</span>
        </div>
        <div class="field">
          <label for="backend">Transcription engine</label>
          <select name="backend" id="backend">
            <option value="whispermlx" {% if default_backend == "whispermlx" %}selected{% endif %}>Whisper MLX — includes speaker diarization</option>
            <option value="faster-whisper" {% if default_backend == "faster-whisper" %}selected{% endif %}>Faster Whisper — no speaker diarization</option>
          </select>
          <span class="hint">Whisper MLX requires an HF_TOKEN configured on the server.</span>
        </div>
        <button type="submit" id="submit-button"><span class="button-icon" aria-hidden="true">↗</span><span>Transcribe audio</span></button>
      </form>
    </section>
    <div class="status" id="status" role="status" aria-live="polite"></div>
    <div class="progress" id="progress" hidden>
      <div class="progress-label" id="progress-label">Preparing transcription…</div>
      <div class="progress-track" role="progressbar" aria-label="Transcription progress"
           aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
        <div class="progress-value" id="progress-value"></div>
      </div>
    </div>
    <section class="result-wrap" id="result-wrap" aria-label="Transcription result" hidden>
      <h2 class="result-heading">Merged transcript · JSON</h2>
      <pre id="result"></pre>
    </section>
    <p class="footnote">Processing runs on the server and may take a few minutes.</p>
  </main>
  <script>
    const f = document.getElementById('f');
    const button = document.getElementById('submit-button');
    const backend = document.getElementById('backend');
    const progress = document.getElementById('progress');
    const progressLabel = document.getElementById('progress-label');
    const progressTrack = progress.querySelector('[role="progressbar"]');
    const progressValue = document.getElementById('progress-value');

    async function watchProgress(progressId, shouldStop) {
      while (!shouldStop()) {
        try {
          const response = await fetch('/progress/' + encodeURIComponent(progressId));
          if (response.ok) {
            const update = await response.json();
            if (shouldStop()) return;
            progressLabel.textContent = update.message;
            progressTrack.setAttribute('aria-valuenow', update.percentage);
            progressValue.style.width = update.percentage + '%';
            if (update.state !== 'working') return;
          } else if (response.status !== 404) {
            throw new Error('HTTP ' + response.status);
          }
        } catch (err) {
          if (!shouldStop()) progressLabel.textContent = 'Unable to retrieve transcription progress.';
          return;
        }
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    }

    f.addEventListener('submit', async (e) => {
      e.preventDefault();
      const status = document.getElementById('status');
      const result = document.getElementById('result');
      const resultWrap = document.getElementById('result-wrap');
      resultWrap.hidden = true;
      status.className = 'status visible';
      status.dataset.state = 'working';
      status.textContent = 'Uploading audio and transcribing both language passes. This may take a few minutes…';
      button.disabled = true;
      const fd = new FormData(f);
      let stopProgress = false;
      const showProgress = backend.value === 'faster-whisper';
      progress.hidden = !showProgress;
      if (showProgress) {
        const progressId = crypto.randomUUID();
        fd.append('progress_id', progressId);
        progressLabel.textContent = 'Preparing transcription…';
        progressTrack.setAttribute('aria-valuenow', '0');
        progressValue.style.width = '0%';
        watchProgress(progressId, () => stopProgress);
      }
      try {
        const r = await fetch('/transcribe', { method: 'POST', body: fd });
        const j = await r.json();
        if (!r.ok) throw new Error(j.error || ('HTTP ' + r.status));
        stopProgress = true;
        status.dataset.state = 'success';
        status.textContent = 'Transcription complete with ' + j.backend + '. ' + j.segment_count +
          ' segments saved to ' + j.merged_file +
          (j.deduped_count ? ' (' + j.deduped_count + ' repeated English segments removed).' : '.');
        result.textContent = JSON.stringify(j.result, null, 2);
        resultWrap.hidden = false;
      } catch (err) {
        stopProgress = true;
        status.dataset.state = 'error';
        status.textContent = 'Error: ' + err.message;
      } finally {
        stopProgress = true;
        progress.hidden = true;
        button.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML, default_backend=DEFAULT_BACKEND)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "hf_token_set": bool(HF_TOKEN),
        "device": DEVICE,
        "model": MODEL,
        "default_backend": DEFAULT_BACKEND,
        "backends": BACKENDS,
        "transcriptions_dir": str(TRANSCRIPTIONS_DIR),
        "dedup_sim_threshold": SIM_THRESHOLD,
    })


@app.route("/progress/<progress_id>", methods=["GET"])
def transcription_progress(progress_id):
    try:
        progress_id = str(uuid.UUID(progress_id))
    except ValueError:
        return jsonify({"error": "invalid progress id"}), 400
    progress = _get_transcription_progress(progress_id)
    if progress is None:
        return jsonify({"error": "progress not found"}), 404
    return jsonify(progress)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "missing 'audio' file in multipart form"}), 400

    upload = request.files["audio"]
    if not upload.filename:
        return jsonify({"error": "empty filename"}), 400

    backend = request.form.get("backend", DEFAULT_BACKEND).strip().lower()
    if backend not in BACKENDS:
        return jsonify({"error": f"backend must be one of: {', '.join(BACKENDS)}"}), 400

    progress_id = None
    if backend == "faster-whisper":
        requested_progress_id = request.form.get("progress_id")
        if requested_progress_id:
            try:
                progress_id = str(uuid.UUID(requested_progress_id))
            except ValueError:
                return jsonify({"error": "invalid progress id"}), 400
            _set_transcription_progress(progress_id, 0, "Preparing transcription…")

    basename = Path(upload.filename).name
    stem = re.sub(r"\.m4a$", "", basename, flags=re.IGNORECASE)
    stem = Path(stem).stem or f"upload_{uuid.uuid4().hex[:8]}"

    workdir = Path(tempfile.mkdtemp(prefix="transcribe_"))
    audio_path = workdir / basename
    upload.save(audio_path)

    try:
        for language, output_dir, start_percentage in (
            ("ro", RO_DIR, 0),
            ("en", EN_DIR, 50),
        ):
            language_name = "Romanian" if language == "ro" else "English"
            if progress_id:
                _set_transcription_progress(
                    progress_id, start_percentage, f"Transcribing {language_name} pass…"
                )

            def report_pass_progress(ratio):
                if progress_id:
                    _set_transcription_progress(
                        progress_id,
                        start_percentage + max(0, min(1, ratio)) * 45,
                        f"Transcribing {language_name} pass…",
                    )

            if progress_id:
                run_whisper(
                    audio_path, output_dir, language, backend,
                    progress_callback=report_pass_progress,
                )
            else:
                run_whisper(audio_path, output_dir, language, backend)

        ro_file = RO_DIR / f"{stem}.json"
        en_file = EN_DIR / f"{stem}.json"
        merged_file = TRANSCRIPTIONS_DIR / f"{stem}.merged.json"

        if not ro_file.exists():
            raise RuntimeError(f"expected Romanian output not found: {ro_file}")
        if not en_file.exists():
            raise RuntimeError(f"expected English output not found: {en_file}")

        with open(ro_file, encoding="utf-8") as f:
            ro_data = json.load(f)
        with open(en_file, encoding="utf-8") as f:
            en_data = json.load(f)

        result = merge_transcriptions(ro_data, en_data, str(merged_file))

        with open(merged_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        if progress_id:
            _set_transcription_progress(progress_id, 100, "Transcription complete.", "completed")
        return jsonify({
            "backend": backend,
            "ro_file": str(ro_file),
            "en_file": str(en_file),
            "merged_file": str(merged_file),
            "segment_count": len(result["segments"]),
            "deduped_count": result.get("deduped_count", 0),
            "result": result,
        })

    except Exception as e:
        if progress_id:
            _set_transcription_progress(progress_id, 0, str(e), "error")
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@app.route("/download/<path:filename>", methods=["GET"])
def download(filename):
    # Restrict to transcriptions dir
    target = (TRANSCRIPTIONS_DIR / filename).resolve()
    root = TRANSCRIPTIONS_DIR.resolve()
    if root not in target.parents and target != root:
        return jsonify({"error": "invalid path"}), 400
    if not target.exists():
        return jsonify({"error": "not found"}), 404
    return send_file(target, as_attachment=True)


if __name__ == "__main__":
    # Development server; use gunicorn/waitress in production
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5002)), debug=False)
