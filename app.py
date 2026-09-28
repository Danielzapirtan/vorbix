import os
import re
import json
import shutil
import subprocess
import tempfile
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


def faster_whisper_transcribe(audio_path, output_dir, language):
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
    data = {
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "avg_logprob": segment.avg_logprob,
                "no_speech_prob": segment.no_speech_prob,
                "compression_ratio": segment.compression_ratio,
                "speaker": None,
            }
            for segment in segments
        ],
    }
    output_file = output_dir / f"{Path(audio_path).stem}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data


def run_whisper(audio_path, output_dir, language, backend):
    if backend == "faster-whisper":
        return faster_whisper_transcribe(audio_path, output_dir, language)

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
  <title>Transcription Service</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 780px; margin: 2rem auto; padding: 0 1rem; }
    input[type=file] { margin: 1rem 0; }
    button { padding: 0.6rem 1.2rem; font-size: 1rem; cursor: pointer; }
    pre { background: #f4f4f4; padding: 1rem; overflow: auto; max-height: 60vh; }
    .status { margin-top: 1rem; color: #444; }
  </style>
</head>
<body>
  <h1>Audio → Bilingual Transcript</h1>
  <p>Runs the selected backend twice (Romanian + English) and merges the results. Speaker diarization is available with whispermlx.</p>
  <form id="f">
    <input type="file" name="audio" accept="audio/*,.m4a" required>
    <br>
    <label for="backend">Transcription backend:</label>
    <select name="backend" id="backend">
      <option value="whispermlx" {% if default_backend == "whispermlx" %}selected{% endif %}>whispermlx (speaker diarization)</option>
      <option value="faster-whisper" {% if default_backend == "faster-whisper" %}selected{% endif %}>Faster Whisper (no speaker diarization)</option>
    </select>
    <br>
    <button type="submit">Transcribe</button>
  </form>
  <div class="status" id="status"></div>
  <pre id="result" style="display:none"></pre>
  <script>
    const f = document.getElementById('f');
    f.addEventListener('submit', async (e) => {
      e.preventDefault();
      const status = document.getElementById('status');
      const result = document.getElementById('result');
      result.style.display = 'none';
      status.textContent = 'Uploading & transcribing… (this can take a while)';
      const fd = new FormData(f);
      try {
        const r = await fetch('/transcribe', { method: 'POST', body: fd });
        const j = await r.json();
        if (!r.ok) throw new Error(j.error || ('HTTP ' + r.status));
        status.textContent = 'Done with ' + j.backend + '. Merged file: ' + j.merged_file +
          (j.deduped_count ? ' (deduped ' + j.deduped_count + ' EN segment(s))' : '');
        result.textContent = JSON.stringify(j.result, null, 2);
        result.style.display = 'block';
      } catch (err) {
        status.textContent = 'Error: ' + err.message;
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

    basename = Path(upload.filename).name
    stem = re.sub(r"\.m4a$", "", basename, flags=re.IGNORECASE)
    stem = Path(stem).stem or f"upload_{uuid.uuid4().hex[:8]}"

    workdir = Path(tempfile.mkdtemp(prefix="transcribe_"))
    audio_path = workdir / basename
    upload.save(audio_path)

    try:
        run_whisper(audio_path, RO_DIR, "ro", backend)
        run_whisper(audio_path, EN_DIR, "en", backend)

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
