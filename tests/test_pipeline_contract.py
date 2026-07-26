import os
import asyncio
import subprocess
import sys
import unittest
from io import BytesIO
from io import StringIO
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch
from fastapi import BackgroundTasks, HTTPException, UploadFile


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import app
import media_tools
import processor
from runtime_logs import RuntimeLogBroker, runtime_logs


class MediaToolsTests(unittest.TestCase):
    def test_ffmpeg_resolves_to_an_existing_executable(self):
        self.assertTrue(Path(media_tools.get_ffmpeg_executable()).is_file())

    def test_nonempty_file_guard_rejects_missing_file(self):
        with self.assertRaisesRegex(RuntimeError, "was not created"):
            media_tools.require_nonempty_file(str(PROJECT_DIR / "missing-test-file.mp4"), "Test output")


class JobContractTests(unittest.TestCase):
    def setUp(self):
        app.jobs.clear()

    def test_missing_pipeline_output_marks_job_failed(self):
        job_id = "missing-output"
        app.jobs[job_id] = {"status": "processing", "result_url": None, "error": None}
        missing = PROJECT_DIR / "output_videos_wav2lip" / "does-not-exist.mp4"

        with patch.object(app, "run_wav2lip_hd_pipeline", return_value=str(missing)):
            app.process_video_task(job_id, "video.mp4", "audio.wav", False)

        self.assertEqual(app.jobs[job_id]["status"], "failed")
        self.assertIsNone(app.jobs[job_id]["result_url"])
        self.assertIn("was not created", app.jobs[job_id]["error"])

    def test_existing_pipeline_output_is_published_with_url(self):
        job_id = "existing-output"
        output = Path(app.OUTPUT_VIDEO_WAV2LIP_DIR) / f"{job_id}.mp4"
        output.write_bytes(b"test video payload")
        self.addCleanup(output.unlink, missing_ok=True)
        app.jobs[job_id] = {"status": "processing", "result_url": None, "error": "old"}

        with patch.object(app, "run_wav2lip_hd_pipeline", return_value=str(output)):
            app.process_video_task(job_id, "video.mp4", "audio.wav", False)

        self.assertEqual(
            app.jobs[job_id],
            {
                "status": "completed",
                "progress": 100,
                "result_url": f"/outputs_fast/{job_id}.mp4",
                "error": None,
            },
        )

    def test_unsupported_upload_extension_is_rejected(self):
        with self.assertRaisesRegex(Exception, "Unsupported video file type"):
            app._validated_extension("payload.exe", app.VIDEO_EXTENSIONS, "video")

    def test_generate_accepts_nonempty_supported_uploads(self):
        job_id = "00000000-0000-0000-0000-000000000001"
        video_path = Path(app.INPUT_VIDEO_DIR) / f"{job_id}.mp4"
        audio_path = Path(app.INPUT_AUDIO_DIR) / f"{job_id}.wav"
        self.addCleanup(video_path.unlink, missing_ok=True)
        self.addCleanup(audio_path.unlink, missing_ok=True)
        video = UploadFile(filename="face.mp4", file=BytesIO(b"video"))
        audio = UploadFile(filename="speech.wav", file=BytesIO(b"audio"))

        with patch.object(app.uuid, "uuid4", return_value=job_id):
            response = asyncio.run(
                app.generate_video(BackgroundTasks(), video, audio, use_esrgan=False)
            )

        self.assertEqual(response, {"job_id": job_id})
        self.assertEqual(app.jobs[job_id]["status"], "processing")
        self.assertTrue(video_path.exists())
        self.assertTrue(audio_path.exists())

    def test_generate_rejects_empty_uploads(self):
        video = UploadFile(filename="face.mp4", file=BytesIO())
        audio = UploadFile(filename="speech.wav", file=BytesIO(b"audio"))
        with self.assertRaises(HTTPException) as error:
            asyncio.run(app.generate_video(BackgroundTasks(), video, audio, use_esrgan=False))
        self.assertEqual(error.exception.status_code, 400)


class ProcessorTests(unittest.TestCase):
    def test_subprocess_failure_is_reported_as_pipeline_step(self):
        class FakeProcess:
            def __init__(self):
                self.stdout = StringIO("model ready\nERROR synthetic failure\n")

            def wait(self):
                return 7

        with patch.object(processor.subprocess, "Popen", return_value=FakeProcess()):
            with self.assertRaisesRegex(RuntimeError, "step.py.*exit code 7"):
                processor._run(["python", "step.py"], str(PROJECT_DIR))

        captured = runtime_logs.recent(5)
        self.assertTrue(any(entry["message"] == "model ready" for entry in captured))
        self.assertTrue(any(entry["level"] == "error" for entry in captured))

    def test_recovered_video_has_a_valid_frame_rate(self):
        recovered = PROJECT_DIR / "output_videos_wav2lip" / "c9efabdb-b39a-4418-8c61-deb7c696a255.mp4"
        if not recovered.exists():
            self.skipTest("Recovered runtime video is not available")
        self.assertGreater(processor._get_video_fps(str(recovered)), 0)


class RuntimeLogTests(unittest.TestCase):
    def test_broker_bounds_history_and_removes_terminal_codes(self):
        broker = RuntimeLogBroker(capacity=2)
        broker.publish("first", source="test")
        broker.publish("\x1b[31msecond\x1b[0m", level="warning", source="test")
        broker.publish("third", job_id="job-123", source="test")

        entries = broker.recent(10)
        self.assertEqual([entry["message"] for entry in entries], ["second", "third"])
        self.assertEqual(entries[-1]["job_id"], "job-123")

    def test_wait_after_returns_only_newer_entries(self):
        broker = RuntimeLogBroker(capacity=5)
        first = broker.publish("first")
        broker.publish("second")
        self.assertEqual(
            [entry["message"] for entry in broker.wait_after(first["id"], timeout=0)],
            ["second"],
        )


class FrontendTimerTests(unittest.TestCase):
    def test_generation_timer_elements_and_lifecycle_are_present(self):
        class IdCollector(HTMLParser):
            def __init__(self):
                super().__init__()
                self.ids = set()

            def handle_starttag(self, tag, attrs):
                element_id = dict(attrs).get("id")
                if element_id:
                    self.ids.add(element_id)

        parser = IdCollector()
        parser.feed((PROJECT_DIR / "static" / "index.html").read_text(encoding="utf-8"))
        self.assertTrue(
            {"generation-clock", "generation-clock-label", "generation-clock-value"}.issubset(parser.ids)
        )

        script = (PROJECT_DIR / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("startGenerationTimer(runId)", script)
        self.assertIn("stopGenerationTimer('completed', runId)", script)
        self.assertIn("stopGenerationTimer('failed', runId)", script)


if __name__ == "__main__":
    unittest.main()
