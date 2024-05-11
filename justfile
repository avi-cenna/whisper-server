#!/usr/bin/env just --justfile

serve:
  ./run_loop.sh

# send:
#   poetry run python main.py send

# test-transcription:
#   poetry run python main.py test-transcription

fmt:
  poetry run black .
  poetry run isort .
