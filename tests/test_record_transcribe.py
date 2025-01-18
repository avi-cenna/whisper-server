import time
from pathlib import Path
from threading import Event, Thread

import pytest
from loguru import logger

from src.whisper_server.config import load_config
from src.whisper_server.record_transcribe import (
    transcribe,
    transcribe_api,
    try_record_audio,
)

SAMPLE_MODELS = [
    "mlx-community/whisper-large-v3-turbo",
    "mlx-community/whisper-large-v3-turbo-q4",
    "mlx-community/whisper-tiny.en-mlx",
]

@pytest.fixture
def config():
    return load_config()

@pytest.fixture
def audio_recording(config):
    """Create a test audio recording"""
    stop_event = Event()
    
    def record():
        logger.warning("Recording audio")
        return try_record_audio(config, stop_event)
    
    recording_thread = Thread(target=record)
    recording_thread.start()
    
    # Record for 5 seconds
    time.sleep(5)
    stop_event.set()
    recording_thread.join()

    # Wait for recording to complete and get the path
    recording_path = recording_thread.join()
    assert Path(recording_path).is_file()
    
    yield recording_path
    # Cleanup
    recording_path.unlink(missing_ok=True)

def test_record_and_output_file(audio_recording):
    """Test that audio recording creates a valid file"""
    assert audio_recording.stat().st_size > 0


@pytest.mark.slow
@pytest.mark.parametrize("model", SAMPLE_MODELS)
def test_transcribe_with_models(config, audio_recording, model):
    """Test transcription with different whisper models"""
    config.whisper_model_config.model = model
    result = transcribe(audio_recording, config)
    assert isinstance(result, str)
    assert len(result) > 0
    logger.info(f"Transcription result for {model}: {result}")

@pytest.mark.slow
@pytest.mark.api
def test_transcribe_api(config, audio_recording):
    """Test transcription using the OpenAI API"""
    result = transcribe_api(audio_recording, config)
    assert isinstance(result, str)
    assert len(result) > 0
    logger.info(f"API transcription result: {result}")
