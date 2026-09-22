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
else
	whispermlx --help
	whispermlx "$AUDIO_FILE" --compression_ratio_threshold 2.4 --condition_on_previous_text False --device $DEVICE --diarize --hf_token $HF_TOKEN --language ro --logprob_threshold -1.0 --model large-v3 --no_speech_threshold 0.6 --output_dir $odir --output_format json
	ofile=$odir/$(basename $AUDIO_FILE | sed -e "s/\.m4a//g").json
	echo "=== Transcription"
	jq . $ofile
	echo "==="
fi

