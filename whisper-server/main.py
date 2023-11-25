from pathlib import Path

import client
import server
import typer

from typer import Argument, Option, Typer
from typing_extensions import Annotated

app = Typer()


@app.command()
def serve(
    auto_stop: Annotated[bool, Argument(help="If True, the server will stop automatically")] = True
):
    """Serve the Whisper transcription service, receiving ZMQ messages and returning transcriptions."""
    server.main()


@app.command()
def send():
    """Send a message to the Whisper server."""
    client.main()


@app.command()
def test_transcription():
    """Test the transcription service."""
    infile = Path("~/TODO/jfk.wav").expanduser()
    server.fw(infile.as_posix())


if __name__ == "__main__":
    app()
