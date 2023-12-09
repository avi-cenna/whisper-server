#!/usr/bin/env just --justfile

serve:
  poetry run python main.py serve

send:
  poetry run python main.py send

# test-transcription:
#   poetry run python main.py test-transcription

fmt:
  poetry run black .
  poetry run isort .
