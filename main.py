from pathlib import Path

from loguru import logger
from typer import Argument, Option, Typer
from typing_extensions import Annotated

app = Typer()


@app.command()
def serve():
    # import whisper_server.server as server

    """Serve the Whisper transcription service, receiving ZMQ messages and returning transcriptions."""
    from whisper_server import server

    server.main()


@app.command()
def send():
    """Send a message to the Whisper server."""
    from whisper_server import client

    client.main()


# TODO: uncomment this
# @app.command()
# def test_transcription():
#     """Test the transcription service."""
#     infile = Path("~/TODO/jfk.wav").expanduser()
#     import whisper_server.config as config
#     import whisper_server.server as server
#     server.transcribe(infile, config.load_config())


if __name__ == "__main__":
    app()
