import queue
import threading
import time
from enum import Enum
from pathlib import Path

import mlx_whisper
from pydantic import BaseModel
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from loguru import logger

from src.whisper_server.autocorrect import autocorrect
from src.whisper_server.config import WhisperServerConfig, load_config
from src.whisper_server.record_transcribe import (
    record_audio,
    transcribe,
    try_record_audio,
)

stop_recording_event = threading.Event()
cfg = load_config()
logger.debug(cfg)
result_queue = queue.Queue()

app = FastAPI()

# TODO: adjust this so that it's an enum class
SAMPLE_MODELS = [
    "mlx-community/whisper-large-v3-turbo",
    "mlx-community/whisper-large-v3-turbo-q4",
    "mlx-community/whisper-tiny.en-mlx",
]

# def thread_record_dep(config: WhisperServerConfig, result_queue):
#     try:
#         result = record_audio(config, stop_recording_event)
#         result_queue.put(result)
#     except Exception as e:
#         result_queue.put(e)


def thread_record(config: WhisperServerConfig):
    result = try_record_audio(config, stop_recording_event)
    result_queue.put(result)


@app.get("/start")
async def start_recording(background_tasks: BackgroundTasks):
    logger.debug(f"{stop_recording_event=}")
    try:
        if stop_recording_event.is_set():
            raise Exception("Recording already in progress")
        stop_recording_event.clear()
        background_tasks.add_task(thread_record, cfg)
        # background_tasks.add_task(thread_record, cfg, result_queue)
        logger.debug("Started recording")
        return {"status": "recording started"}
    except Exception as e:
        logger.error(f"Failed to start recording: {e}")
        stop_recording_event.clear()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stop")
async def stop_recording():
    logger.debug(f"{stop_recording_event=}")
    try:
        logger.debug(f"Stopping recording")
        stop_recording_event.set()

        wavfile = result_queue.get(timeout=10)
        if isinstance(wavfile, Exception):
            raise wavfile
        if not wavfile:
            raise Exception("Failed to save recorded audio. Check logs for details.")
        logger.debug(f"Recorded audio is saved at: {wavfile}")

        transcription = transcribe(wavfile, cfg)
        # transcription = autocorrect(transcription)
        # transcription = transcription.strip() + " "
        logger.info(transcription)

        return {"status": "recording stopped", "transcription": "foo"}
        # return {"status": "recording stopped", "transcription": transcription}
    except Exception as e:
        raise e
        logger.error(e.__traceback__)
        logger.error(f"Failed to stop recording: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        stop_recording_event.clear()

class TranscriptionResult(BaseModel):
    text: str
    duration_ms: int

@app.get("/demo")
def demo(model: WhisperModel = WhisperModel.LARGE_V3_TURBO) -> TranscriptionResult:
    logger.debug(f"{stop_recording_event=}")
    ar = Path('./samples/jfk.wav')
    logger.debug("Starting transcription")
    
    start_time = time.time()
    result = mlx_whisper.transcribe(
        ar.as_posix(),
        path_or_hf_repo=model.value,
        initial_prompt='',
        language='en',
    )
    duration_ms = int((time.time() - start_time) * 1000)
    
    text = result["text"]
    logger.debug("Finished transcription")
    logger.info(f'Transcription result for {model}: {text}')
    
    return TranscriptionResult(text=text, duration_ms=duration_ms)


def main():
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
