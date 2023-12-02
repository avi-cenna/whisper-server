import os
import threading
import tempfile
import wave

from pathlib import Path

import numpy as np
import sounddevice as sd
import webrtcvad
import zmq

from faster_whisper import WhisperModel
from loguru import logger


def record(config=None) -> Path:
    sample_rate = 16000
    frame_duration = 30  # 30ms, supported values: 10, 20, 30
    buffer_duration = 300  # 300ms
    silence_duration = config["silence_duration"] if config else 2200  # 900ms

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
        callback=lambda indata, frames, time, status: buffer.extend(indata[:, 0]),
    ):
        while True:
            # logger.debug("Continuing recording")
            if len(buffer) < sample_rate * frame_duration // 1000:
                # logger.debug("Buffer not full")
                continue

            frame = buffer[: sample_rate * frame_duration // 1000]
            buffer = buffer[sample_rate * frame_duration // 1000:]

            is_speech = vad.is_speech(np.array(frame).tobytes(), sample_rate)
            if is_speech:
                logger.debug("Speech detected")
                recording.extend(frame)
                num_silent_frames = 0
            else:
                logger.debug("Silence detected")
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


def transcribe(wavfile: Path):
    # model_size = "medium.en"
    # model_size = "tiny.en"
    model_size = "small.en"
    model = WhisperModel(model_size, compute_type="float32")
    # model_size_or_path: Size of the model to use (tiny, tiny.en, base, base.en,
    #  small, small.en, medium, medium.en, large-v1, large-v2, or large), a path to a converted
    #  model directory, or a CTranslate2-converted Whisper model ID from the Hugging Face Hub.
    #  When a size or a model ID is configured, the converted model is downloaded
    #  from the Hugging Face Hub.
    logger.debug("Starting transcription")
    init = """ """
    segments, info = model.transcribe(wavfile.as_posix(), language="en", initial_prompt=init)
    logger.debug(
        "Detected language '%s' with probability %f" % (info.language, info.language_probability)
    )
    logger.debug("Finished transcription")
    return "".join(s.text for s in segments)


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    logger.debug("Server started")

    while True:
        logger.debug(f"Waiting for message")
        message = socket.recv().decode("utf-8")
        logger.debug(f"Received request: {message}")

        match message:
            case "start":
                wavfile = record()
                tx = transcribe(wavfile)
                logger.info(tx)
                socket.send_string(tx)
            # case "get":
            #     socket.send_string(tx)
            # TODO: add support for stop here
            case _:
                # tx = transcribe(wavfile)
                # socket.send_string(f"Received invalid message: {message}")
                pass


if __name__ == "__main__":
    main()
