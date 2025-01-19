#!/usr/bin/env just --justfile

# serve-loop:
#  ./run_loop.sh

#    LARGE_V3_TURBO = "mlx-community/whisper-large-v3-turbo"
#    LARGE_V3_TURBO_Q4 = "mlx-community/whisper-large-v3-turbo-q4"
#    TINY_EN = "mlx-community/whisper-tiny.en-mlx"
#    BASE_EN_Q4 = "mlx-community/whisper-base.en-mlx-q4"
#    MEDIUM = "mlx-community/whisper-medium-mlx"
#    MEDIUM_Q4 = "mlx-community/whisper-medium-mlx-q4"
#    MEDIUM_FP32 = "mlx-community/whisper-medium-mlx-fp32"
#    MEDIUM_EN = "mlx-community/whisper-medium.en-mlx"
#    MEDIUM_EN_4BIT = "mlx-community/whisper-medium.en-mlx-4bit"
#    MEDIUM_EN_FP32 = "mlx-community/whisper-medium.en-mlx-fp32"
#    MEDIUM_EN_8BIT = "mlx-community/whisper-medium.en-mlx-8bit"

serve:
  uv run python main.py 'serve'

# send:
#   poetry run python main.py send

# test-transcription:
#   poetry run python main.py test-transcription

fmt:
  uv run black .
  uv run isort .

bench:
#TODO: add more models to this hyperfine command
#      - also, make it so that each one is run 4 times.
  hyperfine \
    'curl localhost:8000/demo?model="mlx-community/whisper-large-v3-turbo"' \
    'curl localhost:8000/demo?model="mlx-community/whisper-large-v3-turbo-q4"' \
    'curl localhost:8000/demo?model="mlx-community/whisper-tiny.en-mlx"'
