from pathlib import Path
from pprint import pprint
from typing import List, Literal, Optional, Union

import client
import server
import typer
import yaml

from pydantic import BaseModel
from typer import Argument, Option, Typer
from typing_extensions import Annotated
from loguru import logger

from pathlib import Path


def find_poetry_project_root(start_path='.'):
    """
    Find the root directory of a Poetry project by searching for the 'pyproject.toml' file.

    :param start_path: The starting directory to begin the search. Defaults to the current directory.
    :return: The path to the project root directory, if found; otherwise, None.
    """
    current_dir = Path(start_path).resolve()

    while current_dir != current_dir.parent:
        if (current_dir / 'pyproject.toml').is_file():
            return current_dir
        current_dir = current_dir.parent

    return None


# TODO: Consider adding some of the commented-out attributes below.
class WhisperModelConfig(BaseModel):
    whisper_model_size: Literal[
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large-v1",
        "large-v2",
        "large",
    ]
    device: str = "auto"
    compute_type: str = "default"
    # cpu_threads: int = 0
    # num_workers: int = 1
    # download_root: Optional[str] = None
    # local_files_only: bool = False


# TODO: Consider adding some of the commented-out attributes below.
class TranscriptionConfig(BaseModel):
    language: Optional[str] = None
    # task: str = "transcribe"
    # beam_size: int = 5
    # best_of: int = 5
    # patience: float = 1
    # length_penalty: float = 1
    # repetition_penalty: float = 1
    # no_repeat_ngram_size: int = 0
    # temperature: Union[float, List[float], Tuple[float, ...]] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # compression_ratio_threshold: Optional[float] = 2.4
    # log_prob_threshold: Optional[float] = -1.0
    # no_speech_threshold: Optional[float] = 0.6
    # condition_on_previous_text: bool = True
    # prompt_reset_on_temperature: float = 0.5
    initial_prompt: Optional[str] = None
    # prefix: Optional[str] = None
    # suppress_blank: bool = True
    # suppress_tokens: Optional[List[int]] = [-1]
    # without_timestamps: bool = False
    # max_initial_timestamp: float = 1.0
    # word_timestamps: bool = False
    # prepend_punctuations: str = "\"'“¿([{-"
    # append_punctuations: str = "\"'.。,，!！?？:：”)]}、"
    # vad_filter: bool = False
    # vad_parameters: Optional[Union[dict, VadOptions]] = None


class Config(BaseModel):
    silence_duration: int
    whisper_model_config: WhisperModelConfig
    transcription_config: TranscriptionConfig


def load_config() -> Config:
    def parse_yml_config(p: Path):
        return Config.model_validate(yaml.safe_load(p.read_text()))

    project_dir = find_poetry_project_root(__file__)
    default_config_file = project_dir / 'resources' / 'default_config.yml'
    user_config_file = project_dir / 'resources' / 'user_config.yml'

    if user_config_file.exists():
        logger.debug(f"Loading user config from {user_config_file}")
        return parse_yml_config(user_config_file)
    else:
        logger.debug(f"Loading default config from {default_config_file}")
        return parse_yml_config(default_config_file)


if __name__ == "__main__":
    c = load_config()
    pprint(c.whisper_model_config)
    pass
