import time
from pathlib import Path
from threading import Event, Thread

import mlx_whisper
import pytest
from loguru import logger

from src.whisper_server.config import WhisperServerConfig, load_config
from src.whisper_server.record_transcribe import (
    transcribe,
    transcribe_api,
    try_record_audio,
)
from whisper_server.whisper_backend import get_backend_class

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
    result: Path | None = None

    def record():
        logger.warning("Recording audio")
        # TODO: fix this warning
        # Shadows name 'result' from outer scope
        # Local variable 'result' value is not used
        result = try_record_audio(config, stop_event)

    recording_thread = Thread(target=record)
    recording_thread.start()

    # Record for 5 seconds
    time.sleep(5)
    stop_event.set()
    recording_thread.join()

    # Wait for recording to complete and get the path
    recording_path = result
    assert Path(recording_path).is_file()

    yield recording_path
    # Cleanup
    recording_path.unlink(missing_ok=True)


@pytest.mark.skip
def test_record_and_output_file(audio_recording):
    """Test that audio recording creates a valid file"""
    assert audio_recording.stat().st_size > 0


@pytest.mark.parametrize("model", SAMPLE_MODELS)
def test_transcribe_with_models(config, model):
    """Test transcription with different whisper models"""
    logger.debug(f"Starting transcription with model: {model}")
    ar = Path("./samples/jfk.wav")
    result = ttranscribe_local(ar, model)
    assert isinstance(result, str)
    assert len(result) > 0
    logger.info(f"Transcription result for {model}: {result}")


def test_transcribe_with_faster_whisper(config):
    """Test transcription with different whisper models"""
    model_name = config.faster_whisper_config.model
    logger.debug(f"Starting transcription with model: {model_name}")
    wavfile = find_project_root() / Path("samples/jfk.wav")
    assert wavfile.is_file()
    backend = get_backend_class("faster-whisper")(config)
    result = backend.transcribe(wavfile)
    assert isinstance(result, str)
    assert len(result) > 0
    logger.info(f"Transcription result for {model_name}: {result}")


@pytest.mark.skip
def test_transcribe_api(config, audio_recording):
    """Test transcription using the OpenAI API"""
    result = transcribe_api(audio_recording, config)
    assert isinstance(result, str)
    assert len(result) > 0
    logger.info(f"API transcription result: {result}")


def ttranscribe_local(wavfile: Path, model: str) -> str:
    logger.debug("Starting transcription")
    result = mlx_whisper.transcribe(
        wavfile.as_posix(),
        path_or_hf_repo=model,
        # language=transciption_cfg.language,
        initial_prompt="",
        language="en",
    )
    result = result["text"]
    logger.debug("Finished transcription")
    print(result)
    return result


def find_project_root(start_path="."):
    """
    Find the root directory of a Poetry project by searching for the 'pyproject.toml' file.

    :param start_path: The starting directory to begin the search. Defaults to the current directory.
    :return: The path to the project root directory, if found; otherwise, None.
    """
    current_dir = Path(start_path).resolve()

    while current_dir != current_dir.parent:
        if (current_dir / "pyproject.toml").is_file():
            return current_dir
        current_dir = current_dir.parent

    return None
