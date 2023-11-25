#!/usr/bin/env just --justfile

serve:
  poetry run python fw/main.py serve

send:
  poetry run python fw/main.py send

test-transcription:
  poetry run python fw/main.py test-transcription

fmt:
  poetry run black .
  poetry run isort .
