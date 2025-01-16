import queue
import threading

import uvicorn

from fastapi import BackgroundTasks, FastAPI, HTTPException
from loguru import logger

from src.whisper_server.autocorrect import autocorrect
from src.whisper_server.config import WhisperServerConfig, load_config
from src.whisper_server.record_transcribe import record_audio, transcribe, try_record_audio

stop_recording_event = threading.Event()
cfg = load_config()
logger.debug(cfg)
result_queue = queue.Queue()

app = FastAPI()


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
        transcription = autocorrect(transcription)
        transcription = transcription.strip() + " "
        logger.info(transcription)
        return {"status": "recording stopped", "transcription": transcription}
    except Exception as e:
        raise e
        logger.error(e.__traceback__)
        logger.error(f"Failed to stop recording: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        stop_recording_event.clear()


def main():
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
