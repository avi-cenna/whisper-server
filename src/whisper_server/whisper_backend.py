import importlib
import tempfile
import time
import traceback
import wave
from pathlib import Path
from random import random
from threading import Event

import mlx_whisper
import numpy as np
import openai
import sounddevice as sd
import webrtcvad

# from whisperplus.
from loguru import logger

from src.whisper_server.config import WhisperServerConfig

# and i want to try this model
# mlx-community/whisper-large-v3-turbo-q4

# from whisperplus.pipelines import mlx_whisper
# from whisperplus import download_youtube_to_mp3
#
# url = "https://www.youtube.com/watch?v=1__CAdTJ5JU"
# audio_path = download_youtube_to_mp3(url)
#
# text = mlx_whisper.transcribe(
#     audio_path, path_or_hf_repo="mlx-community/whisper-large-v3-mlx"
# )["text"]
# print(text)


# Initialize the whisper model
# def initialize_model():
#     return mlx_whisper(path_or_hf_repo="mlx-community/whisper-large-v3-turbo-q4")

# static_model = initialize_model()


# def try_record_audio(config: WhisperServerConfig, stop_recording_event: Event, attempt=0) -> Path | None:
def try_record_audio(
    config: WhisperServerConfig, stop_recording_event: Event, attempt=0
):
    """Attempt to record an audio file and return the path to the file, or None if recording failed."""
    try:
        return record_audio(config, stop_recording_event)
    except sd.PortAudioError as e:
        logger.error(f"Failed to record audio due to PortAudio error: {e}")
        traceback.print_tb(e.__traceback__)
        _refresh_port_audio()
        if attempt == 0:
            return try_record_audio(config, stop_recording_event, attempt=1)
    except Exception as e:
        logger.error(f"Failed to record audio: {e}")


def record_audio(config: WhisperServerConfig, stop_recording_event: Event) -> Path:
    """Record an audio file from system input and return the path to the file."""
    sample_rate = 16000
    frame_duration = 30  # 30ms, supported values: 10, 20, 30
    buffer_duration = 300  # 300ms
    silence_duration = config.silence_duration

    vad = webrtcvad.Vad(3)  # Aggressiveness mode: 3 (highest)
    buffer = []
    recording = []
    num_silent_frames = 0
    num_buffer_frames = buffer_duration // frame_duration
    silence_frames_threshold = silence_duration // frame_duration

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocksize=sample_rate * frame_duration // 1000,
        callback=lambda indata, frames, _time, status: buffer.extend(indata[:, 0]),
    ):
        while True and not stop_recording_event.is_set():
            # logger.debug("Continuing recording")
            if len(buffer) < sample_rate * frame_duration // 1000:
                # logger.debug("Buffer not full")
                continue

            frame = buffer[: sample_rate * frame_duration // 1000]
            buffer = buffer[sample_rate * frame_duration // 1000 :]

            # is_speech = vad.is_speech(np.array(frame).tobytes(), sample_rate)
            is_speech = vad.is_speech(np.array(frame).tobytes(), sample_rate) or True
            if is_speech:
                rand_debug(0.05, "Speech detected")
                recording.extend(frame)
                num_silent_frames = 0
            else:
                rand_debug(0.15, "Silence detected")
                if len(recording) > 0:
                    num_silent_frames += 1

                if num_silent_frames >= silence_frames_threshold:
                    break

    audio_data = np.array(recording, dtype=np.int16)

    # Save the recorded audio as a temporary WAV file on disk
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        with wave.open(temp_audio_file.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes (16 bits) per sample
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        return Path(temp_audio_file.name)


def _refresh_port_audio():
    logger.debug(f"Refreshing PortAudio")
    logger.debug(f"sound devices = {sd.query_devices()}")
    sd._terminate()
    sd._initialize()
    logger.debug(f"sound devices = {sd.query_devices()}")


def transcribe(wavfile: Path, config: WhisperServerConfig) -> str:
    """Transcribe an audio file and return the transcription."""
    start = time.perf_counter()

    if config.local or True:
        logger.debug(f"Using local transcription")
        result = transcribe_local(wavfile, config)
    else:
        result = transcribe_api(wavfile, config)
    logger.success(result)

    end = time.perf_counter()
    logger.debug(f"Finished transcription in {end - start:.2f} seconds.")
    # logger.debug(f"Finished transcription in {end - start:.2f} seconds. duration={info.duration:.2f}")
    return result


def transcribe_api(wavfile: Path, config: WhisperServerConfig) -> str:
    result = openai.audio.transcriptions.create(
        file=wavfile,
        model="whisper-1",
        language="en",
        response_format="text",
        prompt=config.transcription_config.initial_prompt,
    )
    return str(result)


def transcribe_local(wavfile: Path, config: WhisperServerConfig) -> str:
    import mlx_whisper

    model_cfg = config.whisper_model_config
    # model = static_model
    logger.debug("Starting transcription")
    transciption_cfg = config.transcription_config
    result = mlx_whisper.transcribe(
        wavfile.as_posix(),
        # path_or_hf_repo='mlx-community/whisper-large-v3-turbo-q4',
        # path_or_hf_repo='mlx-community/whisper-tiny.en-mlx',
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
        # language=transciption_cfg.language,
        initial_prompt=transciption_cfg.initial_prompt,
    )
    result = result["text"]
    logger.debug("Finished transcription")
    print(result)
    return result


def transcribe_local_old(wavfile: Path, config: WhisperServerConfig) -> str:
    import mlx_whisper

    model_cfg = config.whisper_model_config
    # model = static_model
    logger.debug("Starting transcription")
    transciption_cfg = config.transcription_config
    segments, info = mlx_whisper.transcribe(
        wavfile.as_posix(),
        # language=transciption_cfg.language,
        initial_prompt=transciption_cfg.initial_prompt,
    )
    logger.debug(
        "Detected language '%s' with probability %f"
        % (info.language, info.language_probability)
    )
    segments = list(segments)
    result = "".join(s.text for s in segments)
    logger.debug("Finished transcription")
    return result


def rand_debug(pct: float, msg: str):
    """Log a message with a given probability."""
    if random() < pct:
        logger.debug(msg)
