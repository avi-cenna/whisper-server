#!/usr/bin/env just --justfile

serve:
  poetry run python whisper-server/main.py serve

send:
  poetry run python whisper-server/main.py send

test-transcription:
  poetry run python whisper-server/main.py test-transcription

fmt:
  poetry run black .
  poetry run isort .
