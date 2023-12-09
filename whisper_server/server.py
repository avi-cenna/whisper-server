import os
import tempfile
import threading
import wave

from pathlib import Path

import numpy as np
import sounddevice as sd
import webrtcvad
import zmq

from whisper_server.config import WhisperServerConfig, load_config
from faster_whisper import WhisperModel
from loguru import logger


def record(config: WhisperServerConfig) -> Path:
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
        callback=lambda indata, frames, time, status: buffer.extend(indata[:, 0]),
    ):
        while True:
            # logger.debug("Continuing recording")
            if len(buffer) < sample_rate * frame_duration // 1000:
                # logger.debug("Buffer not full")
                continue

            frame = buffer[: sample_rate * frame_duration // 1000]
            buffer = buffer[sample_rate * frame_duration // 1000 :]

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


def transcribe(wavfile: Path, config: WhisperServerConfig):
    model_cfg = config.whisper_model_config
    model = WhisperModel(
        model_cfg.whisper_model_size,
        device=model_cfg.device,
        compute_type=model_cfg.compute_type,
    )

    logger.debug("Starting transcription")
    transciption_cfg = config.transcription_config
    segments, info = model.transcribe(
        wavfile.as_posix(),
        language=transciption_cfg.language,
        initial_prompt=transciption_cfg.initial_prompt,
    )
    logger.debug(
        "Detected language '%s' with probability %f"
        % (info.language, info.language_probability)
    )
    logger.debug("Finished transcription")
    return "".join(s.text for s in segments)


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    logger.debug("Server started")

    cfg = load_config()
    logger.debug(cfg)

    while True:
        logger.debug(f"Waiting for message")
        message = socket.recv().decode("utf-8")
        logger.debug(f"Received request: {message}")

        match message:
            case "start":
                wavfile = record(cfg)
                tx = transcribe(wavfile, cfg)
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
