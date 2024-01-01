import os
import queue
import tempfile
import threading
import wave

from pathlib import Path

import numpy as np
import sounddevice as sd
import webrtcvad
import zmq

from faster_whisper import WhisperModel
from loguru import logger

from whisper_server.config import WhisperServerConfig, load_config

_stop_recording_event = threading.Event()


def record(config: WhisperServerConfig) -> Path:
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
        callback=lambda indata, frames, time, status: buffer.extend(indata[:, 0]),
    ):
        while True and not _stop_recording_event.is_set():
            # logger.debug("Continuing recording")
            if len(buffer) < sample_rate * frame_duration // 1000:
                # logger.debug("Buffer not full")
                continue

            frame = buffer[: sample_rate * frame_duration // 1000]
            buffer = buffer[sample_rate * frame_duration // 1000:]

            # is_speech = vad.is_speech(np.array(frame).tobytes(), sample_rate)
            is_speech = vad.is_speech(np.array(frame).tobytes(), sample_rate) or True
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


def transcribe(wavfile: Path, config: WhisperServerConfig) -> str:
    """Transcribe an audio file and return the transcription."""
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
        "Detected language '%s' with probability %f" % (info.language, info.language_probability)
    )
    logger.debug("Finished transcription")
    return "".join(s.text for s in segments)


def thread_record(config: WhisperServerConfig, result_queue):
    try:
        result = record(config)
        result_queue.put(result)
    except Exception as e:
        result_queue.put(e)


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    logger.debug("Server started")

    cfg = load_config()
    logger.debug(cfg)

    record_thread = None
    result_queue = queue.Queue()

    while True:
        logger.debug(f"Waiting for message")
        message = socket.recv().decode("utf-8")
        logger.debug(f"Received request: {message}")

        match message:
            case "start":
                _stop_recording_event.clear()
                record_thread = threading.Thread(target=thread_record, args=(cfg, result_queue))
                record_thread.start()
                logger.debug("Started recording")
                socket.send_string("Started recording")

            case "stop":
                logger.debug(f"Stopping recording")
                _stop_recording_event.set()
                record_thread.join()

                wavfile = result_queue.get()
                if isinstance(wavfile, Exception):
                    raise wavfile
                logger.debug(f"Recorded audio is saved at: {wavfile}")

                # wavfile = record(cfg)
                transcription = transcribe(wavfile, cfg)
                logger.info(transcription)
                socket.send_string(transcription)
            # case "get":
            #     socket.send_string(tx)
            case _:
                socket.send_string(f"Received unsupported message: {message}")


if __name__ == "__main__":
    main()
