import unittest
from threading import Event, Thread
from pathlib import Path
import time

from loguru import logger

# Assuming WhisperServerConfig is defined elsewhere and imported correctly
from whisper_server.config import WhisperServerConfig, load_config
from src.whisper_server.record_transcribe import try_record_audio, transcribe
from whisper_server.record_transcribe import transcribe_api


class TestRecordTranscribe(unittest.TestCase):
    def test_record_and_output_file(self):
        config = load_config()
        stop_event = Event()

        def record():
            return try_record_audio(config, stop_event)

        recording_thread = Thread(target=record)
        recording_thread.start()

        time.sleep(5)  # Record for 5 seconds
        stop_event.set()  # Signal to stop recording

        recording_thread.join()  # Wait for the thread to finish

        # Assuming try_record_audio returns a path or None if it fails
        recording_path = record()
        logger.info(f"Recording path: {recording_path}")

        self.assertIsNotNone(recording_path, "Recording path should not be None")
        self.assertTrue(recording_path.is_file(), "Recorded file should exist")
        self.assertGreater(recording_path.stat().st_size, 0, "Recorded file should not be empty")

    def test_2(self):
        self.assertEqual(1, 1)
        p = Path('/var/folders/6w/cj1n3wl15js15p7xrn235cfh0000gp/T/tmp64bzdwic.wav')
        self.assertTrue(p.is_file())
        t = transcribe_api(p, load_config())
        print(t)
        logger.debug(t)

    def test_3(self):
        self.assertEqual(1, 1)
        p = Path('/var/folders/6w/cj1n3wl15js15p7xrn235cfh0000gp/T/tmp64bzdwic.wav')
        self.assertTrue(p.is_file())
        t = transcribe(p, load_config())
        print(t)
        logger.debug(t)


if __name__ == '__main__':
    unittest.main()
