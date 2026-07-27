import os
import subprocess
import threading
from glob import glob

import cv2

from media_tools import get_ffmpeg_executable, require_nonempty_file
from runtime_logs import publish_log

WAV2LIP_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_ESRGAN_DIR = os.path.join(WAV2LIP_DIR, "Real-ESRGAN")
VENV_PYTHON = os.path.join(WAV2LIP_DIR, "venv", "Scripts", "python.exe")
PIPELINE_LOCK = threading.Lock()
PROCESS_LOCK = threading.Lock()
ACTIVE_PROCESSES = {}
CANCELLED_JOBS = set()


class JobCancelledError(RuntimeError):
    """Raised when a user ends a queued or running generation job."""


def is_job_cancelled(job_id):
    with PROCESS_LOCK:
        return job_id in CANCELLED_JOBS


def clear_job_cancellation(job_id):
    with PROCESS_LOCK:
        CANCELLED_JOBS.discard(job_id)


def cancel_job(job_id):
    """Request cancellation and stop the known subprocess tree when present."""
    with PROCESS_LOCK:
        CANCELLED_JOBS.add(job_id)
        process = ACTIVE_PROCESSES.get(job_id)

    if process is None or process.poll() is not None:
        return False

    if os.name == "nt":
        # The virtual-environment launcher spawns the actual Python process on
        # Windows.  taskkill /T ends that child as well, preventing a GPU job
        # from being orphaned after the UI task is ended.
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        process.terminate()
    return True


def _classify_output(message):
    lowered = message.lower()
    if any(token in lowered for token in ("error", "failed", "traceback", "exception")):
        return "error"
    if "warning" in lowered or "deprecated" in lowered:
        return "warning"
    return "output"


def _run(command, cwd, job_id=None, step=None):
    if job_id and is_job_cancelled(job_id):
        raise JobCancelledError("Generation was cancelled before the pipeline started")

    step_name = step or (os.path.basename(command[1]) if len(command) > 1 else os.path.basename(command[0]))
    publish_log(f"Starting {step_name}", source="pipeline", job_id=job_id)
    print(f"[pipeline] Starting {step_name}")
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    if job_id:
        with PROCESS_LOCK:
            ACTIVE_PROCESSES[job_id] = process

    try:
        if process.stdout is not None:
            for output_line in process.stdout:
                message = output_line.strip()
                if not message:
                    continue
                print(message)
                publish_log(
                    message,
                    level=_classify_output(message),
                    source=step_name,
                    job_id=job_id,
                )
            process.stdout.close()

        return_code = process.wait()
    finally:
        if job_id:
            with PROCESS_LOCK:
                if ACTIVE_PROCESSES.get(job_id) is process:
                    ACTIVE_PROCESSES.pop(job_id, None)

    if job_id and is_job_cancelled(job_id):
        raise JobCancelledError("Generation was cancelled by the user")
    if return_code != 0:
        message = f"Pipeline step '{step_name}' failed with exit code {return_code}"
        publish_log(message, level="error", source="pipeline", job_id=job_id)
        raise RuntimeError(message)
    publish_log(f"Finished {step_name}", level="success", source="pipeline", job_id=job_id)


def _get_video_fps(video_path):
    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open generated video: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    if not fps or fps <= 0:
        raise RuntimeError(f"Could not determine frame rate for: {video_path}")
    return fps

def run_wav2lip_hd_pipeline(
    video_path,
    audio_path,
    base_filename,
    use_esrgan=True,
    face_det_batch_size=4,
    wav2lip_batch_size=64,
    resize_factor=1,
):
    if PIPELINE_LOCK.locked():
        publish_log("Waiting for the active GPU job to finish", source="queue", job_id=base_filename)
    with PIPELINE_LOCK:
        publish_log("GPU pipeline lock acquired", source="queue", job_id=base_filename)
        return _run_wav2lip_hd_pipeline(
            video_path,
            audio_path,
            base_filename,
            use_esrgan,
            face_det_batch_size,
            wav2lip_batch_size,
            resize_factor,
        )


def _run_wav2lip_hd_pipeline(
    video_path,
    audio_path,
    base_filename,
    use_esrgan=True,
    face_det_batch_size=4,
    wav2lip_batch_size=64,
    resize_factor=1,
):
    require_nonempty_file(video_path, "Input video")
    require_nonempty_file(audio_path, "Input audio")
    require_nonempty_file(VENV_PYTHON, "Virtual-environment Python executable")
    require_nonempty_file(os.path.join(WAV2LIP_DIR, "inference.py"), "Wav2Lip inference script")
    require_nonempty_file(
        os.path.join(WAV2LIP_DIR, "checkpoints", "wav2lip_gan.pth"),
        "Wav2Lip checkpoint",
    )

    # Setup paths
    input_video_dir = os.path.join(WAV2LIP_DIR, "input_videos")
    input_audio_dir = os.path.join(WAV2LIP_DIR, "input_audios")
    frames_wav2lip_dir = os.path.join(WAV2LIP_DIR, "frames_wav2lip")
    frames_hd_dir = os.path.join(WAV2LIP_DIR, "frames_hd")
    output_videos_wav2lip_dir = os.path.join(WAV2LIP_DIR, "output_videos_wav2lip")
    output_videos_hd_dir = os.path.join(WAV2LIP_DIR, "output_videos_hd")

    os.makedirs(frames_wav2lip_dir, exist_ok=True)
    os.makedirs(frames_hd_dir, exist_ok=True)
    os.makedirs(output_videos_wav2lip_dir, exist_ok=True)
    os.makedirs(output_videos_hd_dir, exist_ok=True)

    wav2lip_output = os.path.join(output_videos_wav2lip_dir, f"{base_filename}.mp4")

    # 1. Run Wav2Lip inference
    publish_log("Running Wav2Lip inference", source="pipeline", job_id=base_filename)
    _run([
        VENV_PYTHON, "inference.py",
        "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
        "--segmentation_path", "checkpoints/face_segmentation.pth",
        "--sr_path", "checkpoints/esrgan_yunying.pth",
        "--face", video_path,
        "--audio", audio_path,
        "--no_sr", "--no_segmentation",
        "--face_det_batch_size", str(face_det_batch_size),
        "--wav2lip_batch_size", str(wav2lip_batch_size),
        "--resize_factor", str(resize_factor),
        "--outfile", wav2lip_output
    ], cwd=WAV2LIP_DIR, job_id=base_filename, step="Wav2Lip")
    require_nonempty_file(wav2lip_output, "Wav2Lip output")

    if not use_esrgan:
        publish_log("HD enhancement disabled; draft output is ready", level="success", source="pipeline", job_id=base_filename)
        return wav2lip_output

    # 2. Extract frames
    publish_log("Extracting frames for enhancement", source="pipeline", job_id=base_filename)
    frames_path = os.path.join(frames_wav2lip_dir, base_filename)
    _run([
        VENV_PYTHON, "video2frames.py",
        "--input_video", wav2lip_output,
        "--frames_path", frames_path
    ], cwd=WAV2LIP_DIR, job_id=base_filename, step="Frame extraction")

    # 3. Super Resolution (Real-ESRGAN)
    publish_log("Running Real-ESRGAN enhancement", source="pipeline", job_id=base_filename)
    require_nonempty_file(
        os.path.join(REAL_ESRGAN_DIR, "inference_realesrgan.py"),
        "Real-ESRGAN inference script",
    )
    frames_hd_output_path = os.path.join(frames_hd_dir, base_filename)
    _run([
        VENV_PYTHON, "inference_realesrgan.py",
        "-n", "RealESRGAN_x4plus",
        "-i", frames_path,
        "--output", frames_hd_output_path,
        "--outscale", "3.5",
        "--face_enhance"
    ], cwd=REAL_ESRGAN_DIR, job_id=base_filename, step="Real-ESRGAN")
    enhanced_frames = glob(os.path.join(frames_hd_output_path, "frame_*_out.jpg"))
    if not enhanced_frames:
        raise RuntimeError("Real-ESRGAN completed without producing enhanced frames")

    # 4. Stitch with ffmpeg
    publish_log("Stitching enhanced frames with ffmpeg", source="pipeline", job_id=base_filename)
    final_output = os.path.join(output_videos_hd_dir, f"{base_filename}.mp4")

    if os.path.exists(final_output):
        os.remove(final_output)

    image_sequence = os.path.join(frames_hd_output_path, "frame_%05d_out.jpg")
    output_fps = _get_video_fps(wav2lip_output)

    _run([
        get_ffmpeg_executable(), "-y", "-framerate", f"{output_fps:.6f}",
        "-i", image_sequence,
        "-i", audio_path,
        "-vcodec", "libx264", "-crf", "25", "-preset", "veryslow",
        "-acodec", "aac", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-shortest",
        final_output
    ], cwd=WAV2LIP_DIR, job_id=base_filename, step="ffmpeg stitch")
    require_nonempty_file(final_output, "HD output")

    publish_log("Pipeline completed successfully", level="success", source="pipeline", job_id=base_filename)
    return final_output
