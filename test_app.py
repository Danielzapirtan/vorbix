import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import app


class FasterWhisperTests(unittest.TestCase):
    def test_index_offers_both_backends(self):
        response = app.app.test_client().get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'value="whispermlx"', response.data)
        self.assertIn(b'value="faster-whisper"', response.data)

    def test_transcription_is_normalized_to_merge_schema(self):
        segment = SimpleNamespace(
            start=0.5,
            end=1.5,
            text=" Bună ziua.",
            avg_logprob=-0.2,
            no_speech_prob=0.01,
            compression_ratio=1.1,
        )
        info = SimpleNamespace(language="ro", language_probability=0.99)
        model = Mock()
        model.transcribe.return_value = ([segment], info)

        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "sample.wav"
            output_dir = Path(directory) / "ro"
            output_dir.mkdir()
            with patch.object(app, "get_faster_whisper_model", return_value=model):
                result = app.faster_whisper_transcribe(audio_path, output_dir, "ro")

            self.assertEqual(result["language"], "ro")
            self.assertEqual(result["segments"][0]["text"], " Bună ziua.")
            self.assertIsNone(result["segments"][0]["speaker"])
            self.assertEqual(
                json.loads((output_dir / "sample.json").read_text(encoding="utf-8")),
                result,
            )
            model.transcribe.assert_called_once_with(
                str(audio_path),
                language="ro",
                beam_size=5,
                condition_on_previous_text=False,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
            )

    def test_transcription_reports_segment_progress(self):
        segments = [
            SimpleNamespace(
                start=0.0, end=2.0, text="First", avg_logprob=-0.2,
                no_speech_prob=0.01, compression_ratio=1.1,
            ),
            SimpleNamespace(
                start=2.0, end=8.0, text="Second", avg_logprob=-0.2,
                no_speech_prob=0.01, compression_ratio=1.1,
            ),
        ]
        info = SimpleNamespace(language="ro", language_probability=0.99, duration=10.0)
        model = Mock()
        model.transcribe.return_value = (iter(segments), info)
        progress_callback = Mock()

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with patch.object(app, "get_faster_whisper_model", return_value=model):
                app.faster_whisper_transcribe(
                    Path(directory) / "sample.wav",
                    output_dir,
                    "ro",
                    progress_callback=progress_callback,
                )

        self.assertEqual(
            [call.args[0] for call in progress_callback.call_args_list],
            [0.2, 0.8, 1],
        )

    def test_transcribe_uses_selected_backend_for_both_language_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ro_dir = root / "ro"
            en_dir = root / "en"
            ro_dir.mkdir()
            en_dir.mkdir()
            calls = []

            progress_id = "eab34871-9258-4724-b4a2-2eef8bfb139e"

            def fake_run_whisper(audio_path, output_dir, language, backend, progress_callback=None):
                calls.append((language, backend))
                if progress_callback:
                    progress_callback(0.5)
                (output_dir / "sample.json").write_text(
                    json.dumps({"segments": []}), encoding="utf-8"
                )

            with (
                patch.object(app, "TRANSCRIPTIONS_DIR", root),
                patch.object(app, "RO_DIR", ro_dir),
                patch.object(app, "EN_DIR", en_dir),
                patch.object(app, "run_whisper", side_effect=fake_run_whisper),
            ):
                client = app.app.test_client()
                response = client.post(
                    "/transcribe",
                    data={
                        "backend": "faster-whisper",
                        "progress_id": progress_id,
                        "audio": (io.BytesIO(b"audio"), "sample.wav"),
                    },
                    content_type="multipart/form-data",
                )

            self.assertEqual(response.status_code, 200, response.get_json())
            self.assertEqual(response.get_json()["backend"], "faster-whisper")
            self.assertEqual(calls, [("ro", "faster-whisper"), ("en", "faster-whisper")])
            progress = client.get(f"/progress/{progress_id}")
            self.assertEqual(progress.status_code, 200)
            self.assertEqual(progress.get_json()["percentage"], 100)
            self.assertEqual(progress.get_json()["state"], "completed")

    def test_unknown_backend_is_rejected(self):
        client = app.app.test_client()
        response = client.post(
            "/transcribe",
            data={"backend": "unknown", "audio": (io.BytesIO(b"audio"), "sample.wav")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
