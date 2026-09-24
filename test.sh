#! /usr/bin/env bash

AUDIO_FILE="$1"
DEVICE=cpu
IVRIT=ivrit-ai/pyannote-speaker-diarization-3.1
ALTPYA=pyannote/speaker-diarization-2.1
PYA=pyannote/speaker-diarization-community-1
odir=$HOME/transcriptions
VER=3.12

if ! command -v python$VER &>/dev/null; then
	brew install python@$VER
fi
mkdir -p $odir
python$VER -m venv venv
source venv/bin/activate
export VIRTUAL_ENV
python$VER -m pip install --upgrade pip
brew install ffmpeg-full
pip install -r requirements.txt

if test -z $HF_TOKEN; then
	echo "exportati HF_TOKEN"
	exit 1
fi

basename_noext=$(basename "$AUDIO_FILE" | sed -e "s/\.m4a//g")
ro_dir=$odir/ro
en_dir=$odir/en
mkdir -p "$ro_dir" "$en_dir"

echo "=== Pass 1: Romanian ==="
whispermlx "$AUDIO_FILE" \
	--compression_ratio_threshold 2.4 \
	--condition_on_previous_text False \
	--device $DEVICE \
	--diarize \
	--hf_token $HF_TOKEN \
	--language ro \
	--logprob_threshold -1.0 \
	--model large-v3 \
	--no_speech_threshold 0.6 \
	--output_dir "$ro_dir" \
	--output_format json

echo "=== Pass 2: English ==="
whispermlx "$AUDIO_FILE" \
	--compression_ratio_threshold 2.4 \
	--condition_on_previous_text False \
	--device $DEVICE \
	--diarize \
	--hf_token $HF_TOKEN \
	--language en \
	--logprob_threshold -1.0 \
	--model large-v3 \
	--no_speech_threshold 0.6 \
	--output_dir "$en_dir" \
	--output_format json

ro_file="$ro_dir/$basename_noext.json"
en_file="$en_dir/$basename_noext.json"
merged_file="$odir/$basename_noext.merged.json"

echo "=== Merging RO + EN segments ==="
python$VER - "$ro_file" "$en_file" "$merged_file" << 'PYEOF'
import json, sys, re
from difflib import SequenceMatcher

ro_file, en_file, out_file = sys.argv[1], sys.argv[2], sys.argv[3]

with open(ro_file, encoding="utf-8") as f:
    ro_data = json.load(f)
with open(en_file, encoding="utf-8") as f:
    en_data = json.load(f)

def get_segments(data):
    # whispermlx JSON: top-level "segments" or nested under "transcription"
    if isinstance(data, dict) and "segments" in data:
        return data["segments"]
    if isinstance(data, dict) and "transcription" in data:
        return data["transcription"]
    if isinstance(data, list):
        return data
    return []

ro_segs = get_segments(ro_data)
en_segs = get_segments(en_data)

# --- Heuristics for picking the language of each segment ---
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

def score(seg, expected_lang):
    """Return a score; higher = more likely this segment's text is in expected_lang."""
    text = (seg.get("text") or "").strip()
    if not text:
        return -1e9

    avg_logprob = seg.get("avg_logprob", seg.get("avg_logprob", -1.0))
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
    """Find candidate segment with biggest time overlap (or nearest start within tol)."""
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

# Anchor on the RO timeline; for each RO segment, find the matching EN segment
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

# Any EN segments with no RO counterpart (rare) get appended by time
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
# If 90%+ of two or more English segments are essentially the same sentence,
# discard all occurrences except the LAST one.

def normalize_text(t):
    """Lowercase, strip punctuation, collapse whitespace for comparison."""
    t = (t or "").lower()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def similarity(a, b):
    """Return similarity ratio (0..1) between two normalized strings."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

SIM_THRESHOLD = 0.90

# Compute indices of English segments only
en_indices = [i for i, s in enumerate(merged) if s.get("lang") == "en"]

# Union-Find to group near-duplicate English segments
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

# Precompute normalized texts
norm = {i: normalize_text(merged[i].get("text")) for i in en_indices}

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
        # Quick length filter: skip obviously different lengths
        la, lb = len(na), len(nb)
        if min(la, lb) / max(la, lb) < SIM_THRESHOLD:
            continue
        if similarity(na, nb) >= SIM_THRESHOLD:
            union(ia, ib)

# Group by root
groups = {}
for i in en_indices:
    r = find(i)
    groups.setdefault(r, []).append(i)

# Determine which indices to drop: within each group with >1 member, keep only the last.
drop_indices = set()
for r, members in groups.items():
    if len(members) > 1:
        members_sorted = sorted(members, key=lambda i: (merged[i].get("start") or 0))
        # Keep only the last occurrence
        for i in members_sorted[:-1]:
            drop_indices.add(i)

if drop_indices:
    merged = [s for i, s in enumerate(merged) if i not in drop_indices]
    print(f"Deduplicated {len(drop_indices)} near-identical English segment(s).")

# Simple turn-based concatenation for convenience
def merge_adjacent(segs):
    out = []
    for s in segs:
        if out and out[-1]["lang"] == s["lang"] and out[-1].get("speaker") == s.get("speaker"):
            out[-1]["text"] += " " + s["text"]
            out[-1]["end"] = s["end"]
        else:
            out.append(dict(s))
    return out

result = {
    "source_file": out_file,
    "segments": merged,
    "turns": merge_adjacent(merged),
    "full_text": " ".join(s["text"] for s in merged if s["text"]),
}

with open(out_file, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"Merged {len(merged)} segments -> {out_file}")
PYEOF

echo "=== Transcription"
jq . "$merged_file"
echo "==="
