#! /bin/bash

VER=3.12

dealocate &>/dev/null || true
rm -rf .venv
python$VER -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements

python$VER app.py
