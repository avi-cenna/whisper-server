import re

from pydantic import BaseModel

_capitalize = [
    "case management",
    "document management",
]

from loguru import logger

from .config import find_project_root


def autocorrect(transcript: str):
    for cap in _capitalize:
        transcript = transcript.replace(cap, cap.title())
    autos = parse_autocorrect_file()
    for a in autos:
        transcript = re.sub(
            r"\b" + a.original + r"\b", a.replacement, transcript, flags=re.I
        )
    return transcript


class Autocorrection(BaseModel):
    original: str
    replacement: str


def parse_autocorrect_file() -> list[Autocorrection]:
    infile = find_project_root() / ".autocorrect"
    if not infile.exists():
        logger.warning(f"No autocorrect file found at {infile}")
        return []

    autocorrections = []
    for line in infile.read_text().splitlines():
        if not line.strip():
            continue
        original, replacement = line.split("->")
        autocorrections.append(
            Autocorrection(original=original.strip(), replacement=replacement.strip())
        )
    return autocorrections
