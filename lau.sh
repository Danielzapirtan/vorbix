#! /bin/bash

VER=3.12

dealocate &>/dev/null || true
rm -rf .venv
python$VER -m venv .venv
source .venv/bin/activate
python$VER -m pip install --upgrade pip
pip install -r requirements.txt

python$VER app.py
