import os
import queue
import tempfile
import threading
import time
import wave

from pathlib import Path

import numpy as np
import openai
import sounddevice as sd
import uvicorn
import webrtcvad

from fastapi import FastAPI, HTTPException
from faster_whisper import WhisperModel
from loguru import logger

from whisper_server.config import WhisperServerConfig, load_config


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
        callback=lambda indata, frames, _time, status: buffer.extend(indata[:, 0]),
    ):
        while True and not stop_recording_event.is_set():
            # logger.debug("Continuing recording")
            if len(buffer) < sample_rate * frame_duration // 1000:
                # logger.debug("Buffer not full")
                continue

            frame = buffer[: sample_rate * frame_duration // 1000]
            buffer = buffer[sample_rate * frame_duration // 1000:]

            # is_speech = vad.is_speech(np.array(frame).tobytes(), sample_rate)
            is_speech = vad.is_speech(np.array(frame).tobytes(), sample_rate) or True
            if is_speech:
                # logger.debug("Speech detected")
                recording.extend(frame)
                num_silent_frames = 0
            else:
                # logger.debug("Silence detected")
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
    start = time.perf_counter()

    if config.local:
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
    segments = list(segments)
    result = "".join(s.text for s in segments)
    return result


def thread_record(config: WhisperServerConfig, result_queue):
    try:
        result = record(config)
        result_queue.put(result)
    except Exception as e:
        result_queue.put(e)


# context = zmq.Context()
# socket = context.socket(zmq.REP)
# socket.bind("tcp://*:5555")
stop_recording_event = threading.Event()
cfg = load_config()
logger.debug(cfg)

record_thread: threading.Thread | None = None
result_queue = queue.Queue()

app = FastAPI()


@app.get("/start")
def start_recording():
    logger.debug(f'{stop_recording_event=}')
    global record_thread, result_queue
    try:
        if record_thread is not None:
            raise Exception("Recording already in progress")
        stop_recording_event.clear()
        record_thread = threading.Thread(target=thread_record, args=(cfg, result_queue))
        record_thread.start()
        logger.debug("Started recording")
        return {"status": "recording started"}
    except Exception as e:
        logger.error(f"Failed to start recording: {e}")
        record_thread = None
        stop_recording_event.clear()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stop")
def stop_recording():
    logger.debug(f'{stop_recording_event=}')
    global record_thread
    try:
        print(record_thread)
        if record_thread is None:
            raise Exception("Recording not in progress")
        logger.debug(f"Stopping recording")
        stop_recording_event.set()
        record_thread.join()

        wavfile = result_queue.get()
        if isinstance(wavfile, Exception):
            raise wavfile
        logger.debug(f"Recorded audio is saved at: {wavfile}")

        transcription = transcribe(wavfile, cfg)
        transcription = transcription.strip() + " "
        logger.info(transcription)
        return {"status": "recording stopped", "transcription": transcription}
    except Exception as e:
        logger.error(f"Failed to stop recording: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        record_thread = None
        stop_recording_event.clear()


@app.get("/set_event")
def set_event():
    global stop_recording_event
    stop_recording_event.set()
    print(stop_recording_event)
    return {"status": "event set"}


@app.get("/clear_event")
def clear_event():
    global stop_recording_event
    stop_recording_event.clear()
    print(stop_recording_event)
    return {"status": "event cleared"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
