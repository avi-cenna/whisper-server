import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Type

from faster_whisper import WhisperModel
from loguru import logger
from pydantic import BaseModel

from src.whisper_server.config import WhisperServerConfig


class WhisperModelName(str, Enum):
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
    def __init__(self, config: WhisperServerConfig):
        pass

    @staticmethod
    @abstractmethod
    def backend_name() -> str:
        pass

    @abstractmethod
    def transcribe(self, wavfile: Path) -> TranscriptionResult:
        pass


def get_backend_class(backend_name: str) -> Type[WhisperBackend]:
    subclasses = WhisperBackend.__subclasses__()
    for subclass in subclasses:
        if subclass.backend_name() == backend_name:
            return subclass
    raise ValueError(f"Unknown backend: {backend_name}")


class MlxWhisperBackend(WhisperBackend):
    def __init__(self, config: WhisperServerConfig):
        self.config = config

    @staticmethod
    def backend_name() -> str:
        return "mlx-whisper"

    def transcribe(self, wavfile: Path) -> TranscriptionResult:
        import mlx_whisper

        logger.debug("Starting transcription")

        # TODO: is perf_counter() or time.time() better here?
        start_time = time.perf_counter()
        result = mlx_whisper.transcribe(
            wavfile.as_posix(),
            path_or_hf_repo=WhisperModelName.MEDIUM_EN,
            initial_prompt="",
            language="en",
        )
        duration_ms = int((time.perf_counter() - start_time) * 1000)

        text = result["text"]
        logger.debug("Finished transcription")

        return TranscriptionResult(text=text, duration_ms=duration_ms)


class FasterWhisperBackend(WhisperBackend):
    def __init__(self, config: WhisperServerConfig):
        self.temp = 1
        self.config = config

    @staticmethod
    def backend_name() -> str:
        return "faster-whisper"

    def transcribe(self, wavfile: Path) -> str:
        # tiny, tiny. en, base, base. en, small, small. en, distil-small. en,
        # medium, medium. en, distil-medium. en, large-v1, large-v2, large-v3, large,
        # distil-large-v2, distil-large-v3, large-v3-turbo, or turbo
        model_cfg = self.config.faster_whisper_config
        model = WhisperModel(
            model_cfg.model,
            device=model_cfg.device,
            compute_type=model_cfg.compute_type,
        )
        logger.debug("Starting transcription")
        segments, info = model.transcribe(
            wavfile.as_posix(),
            language=self.config.language,
            initial_prompt=self.config.initial_prompt,
        )
        logger.debug(
            "Detected language '%s' with probability %f"
            % (info.language, info.language_probability)
        )
        segments = list(segments)
        result = "".join(s.text for s in segments)
        return result


class OpenAIWhisperBackend(WhisperBackend):
    """
    Backend for using the OpenAI API for transcription.
    """

    def __init__(self, config: WhisperServerConfig):
        self.temp = 1
        self.config = config

    @staticmethod
    def backend_name() -> str:
        return "openai-whisper"

    def transcribe(self, wavfile: Path) -> TranscriptionResult:
        # TODO: Implement this
        pass
