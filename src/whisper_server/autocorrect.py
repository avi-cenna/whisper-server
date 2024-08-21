_capitalize = [
    'case management',
    'document management',
]


def autocorrect(transcript: str):
    for cap in _capitalize:
        transcript = transcript.replace(cap, cap.title())
    return transcript
