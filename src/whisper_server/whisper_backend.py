import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

from loguru import logger
from pydantic import BaseModel


class WhisperModel(str, Enum):
    LARGE_V3_TURBO = "mlx-community/whisper-large-v3-turbo"
    LARGE_V3_TURBO_Q4 = "mlx-community/whisper-large-v3-turbo-q4"
    TINY_EN = "mlx-community/whisper-tiny.en-mlx"
    BASE_EN_Q4 = "mlx-community/whisper-base.en-mlx-q4"
    MEDIUM = "mlx-community/whisper-medium-mlx"
    MEDIUM_Q4 = "mlx-community/whisper-medium-mlx-q4"
    MEDIUM_FP32 = "mlx-community/whisper-medium-mlx-fp32"
    MEDIUM_EN = "mlx-community/whisper-medium.en-mlx"
    MEDIUM_EN_4BIT = "mlx-community/whisper-medium.en-mlx-4bit"
    MEDIUM_EN_FP32 = "mlx-community/whisper-medium.en-mlx-fp32"
    MEDIUM_EN_8BIT = "mlx-community/whisper-medium.en-mlx-8bit"


class TranscriptionResult(BaseModel):
    text: str
    duration_ms: int


class WhisperBackend(ABC):
    @abstractmethod
    # maybe add config param later
    def transcribe(self, wavfile: Path) -> TranscriptionResult:
        pass


class MlxWhisperBackend(WhisperBackend):
    def __init__(self):
        self.temp = 1

    def transcribe(self, wavfile: Path) -> TranscriptionResult:
        import mlx_whisper

        logger.debug("Starting transcription")

        # TODO: is perf_counter() or time.time() better here?
        start_time = time.perf_counter()
        result = mlx_whisper.transcribe(
            wavfile.as_posix(),
            path_or_hf_repo=WhisperModel.LARGE_V3_TURBO,
            initial_prompt="",
            language="en",
        )
        duration_ms = int((time.perf_counter() - start_time) * 1000)

        text = result["text"]
        logger.debug("Finished transcription")

        return TranscriptionResult(text=text, duration_ms=duration_ms)


class OpenAIWhisperBackend(WhisperBackend):
    """
    Backend for using the OpenAI API for transcription.
    """

    def transcribe(self, wavfile: Path) -> TranscriptionResult:
        # TODO: Implement this
        pass
