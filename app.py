from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
import logging
import asyncio
import json
from pathlib import Path

# Ensure ffmpeg installed by winget is in the PATH
winget_ffmpeg_path = r"C:\Users\VICTUS\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin"
if winget_ffmpeg_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = winget_ffmpeg_path + os.pathsep + os.environ.get("PATH", "")

from processor import JobCancelledError, cancel_job, run_wav2lip_hd_pipeline
from media_tools import require_nonempty_file
from runtime_logs import publish_log, runtime_logs

logger = logging.getLogger("wav2lip.app")

VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".avi", ".webm", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
FACE_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"}

app = FastAPI(title="Wav2Lip-HD UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
INPUT_VIDEO_DIR = os.path.join(BASE_DIR, "input_videos")
INPUT_IMAGE_DIR = os.path.join(BASE_DIR, "input_images")
INPUT_AUDIO_DIR = os.path.join(BASE_DIR, "input_audios")
OUTPUT_VIDEO_DIR = os.path.join(BASE_DIR, "output_videos_hd")
OUTPUT_VIDEO_WAV2LIP_DIR = os.path.join(BASE_DIR, "output_videos_wav2lip")
FACE_DETECTION_BATCH_SIZES = {1, 2, 4, 8, 16}
GENERATION_BATCH_SIZES = {16, 32, 64}
RESIZE_FACTORS = {1, 2, 4}

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(INPUT_VIDEO_DIR, exist_ok=True)
os.makedirs(INPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(INPUT_AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
os.makedirs(OUTPUT_VIDEO_WAV2LIP_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Mount output videos for streaming
app.mount("/outputs", StaticFiles(directory=OUTPUT_VIDEO_DIR), name="outputs")
app.mount("/outputs_fast", StaticFiles(directory=OUTPUT_VIDEO_WAV2LIP_DIR), name="outputs_fast")

# Global dict to track status (In a real app, use a DB or Redis)
jobs = {}


def _validated_extension(filename: str | None, allowed: set[str], kind: str) -> str:
    extension = Path(filename or "").suffix.lower()
    if extension not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported {kind} file type: {extension or 'none'}")
    return extension

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.on_event("startup")
async def log_startup():
    publish_log("Backend online — waiting for a generation job", level="success", source="server")


@app.get("/api/logs/recent")
async def get_recent_logs(limit: int = Query(250, ge=1, le=1000)):
    return {"entries": runtime_logs.recent(limit)}


@app.get("/api/logs/stream")
async def stream_logs(request: Request):
    try:
        cursor = int(request.headers.get("last-event-id", "0"))
    except ValueError:
        cursor = 0

    async def event_stream():
        nonlocal cursor
        while True:
            entries = await asyncio.to_thread(runtime_logs.wait_after, cursor, 15)
            if await request.is_disconnected():
                break
            if not entries:
                yield ": keepalive\n\n"
                continue
            for entry in entries:
                cursor = entry["id"]
                payload = json.dumps(entry, ensure_ascii=False)
                yield f"id: {cursor}\nevent: log\ndata: {payload}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.post("/api/generate")
async def generate_video(
    background_tasks: BackgroundTasks,
    face: UploadFile = File(...),
    audio: UploadFile = File(...),
    use_esrgan: bool = Form(True),
    face_det_batch_size: int = Form(4),
    wav2lip_batch_size: int = Form(64),
    resize_factor: int = Form(1),
):
    # Direct calls in tests receive FastAPI's Form default wrapper, while
    # HTTP requests receive the parsed primitive value.
    face_det_batch_size = getattr(face_det_batch_size, "default", face_det_batch_size)
    wav2lip_batch_size = getattr(wav2lip_batch_size, "default", wav2lip_batch_size)
    resize_factor = getattr(resize_factor, "default", resize_factor)

    if face_det_batch_size not in FACE_DETECTION_BATCH_SIZES:
        raise HTTPException(status_code=422, detail="Face detection batch size must be 1, 2, 4, 8, or 16")
    if wav2lip_batch_size not in GENERATION_BATCH_SIZES:
        raise HTTPException(status_code=422, detail="Generation batch size must be 16, 32, or 64")
    if resize_factor not in RESIZE_FACTORS:
        raise HTTPException(status_code=422, detail="Resize factor must be 1, 2, or 4")

    job_id = str(uuid.uuid4())
    face_ext = _validated_extension(face.filename, FACE_EXTENSIONS, "video/image")
    audio_ext = _validated_extension(audio.filename, AUDIO_EXTENSIONS, "audio")

    is_image = face_ext in IMAGE_EXTENSIONS
    face_save_dir = INPUT_IMAGE_DIR if is_image else INPUT_VIDEO_DIR
    face_path = os.path.join(face_save_dir, f"{job_id}{face_ext}")
    audio_path = os.path.join(INPUT_AUDIO_DIR, f"{job_id}{audio_ext}")

    try:
        with open(face_path, "wb") as buffer:
            shutil.copyfileobj(face.file, buffer)
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        face_label = "Uploaded image" if is_image else "Uploaded video"
        require_nonempty_file(face_path, face_label)
        require_nonempty_file(audio_path, "Uploaded audio")
    except Exception:
        for uploaded_path in (face_path, audio_path):
            if os.path.isfile(uploaded_path):
                os.remove(uploaded_path)
        raise HTTPException(status_code=400, detail="The uploaded files are empty or could not be saved")

    face_kind = "image" if is_image else "video"
    jobs[job_id] = {"status": "processing", "progress": 0, "result_url": None, "error": None}
    logger.info(
        "Job %s accepted (face: %s, HD enhancement: %s, detector batch: %s, generation batch: %s, resize: %sx)",
        job_id, face_kind, use_esrgan, face_det_batch_size, wav2lip_batch_size, resize_factor,
    )
    publish_log(
        f"Job accepted — face {face_kind} · {'HD enhancement' if use_esrgan else 'draft mode'}",
        source="server",
        job_id=job_id,
    )

    background_tasks.add_task(
        process_video_task, job_id, face_path, audio_path, use_esrgan,
        face_det_batch_size, wav2lip_batch_size, resize_factor,
    )

    return {"job_id": job_id}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_generation(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != "processing":
        return job

    was_running = cancel_job(job_id)
    job.update({
        "status": "cancelled",
        "progress": 0,
        "result_url": None,
        "error": "Generation was ended by the user",
    })
    publish_log(
        "Cancellation requested; stopping the active pipeline" if was_running else "Queued generation cancelled",
        level="warning",
        source="server",
        job_id=job_id,
    )
    return job

def process_video_task(
    job_id: str,
    video_path: str,
    audio_path: str,
    use_esrgan: bool,
    face_det_batch_size: int = 4,
    wav2lip_batch_size: int = 64,
    resize_factor: int = 1,
):
    try:
        # Run the pipeline
        final_video_path = run_wav2lip_hd_pipeline(
            video_path,
            audio_path,
            job_id,
            use_esrgan,
            face_det_batch_size,
            wav2lip_batch_size,
            resize_factor,
        )

        require_nonempty_file(final_video_path, "Generated video")

        # Publish the URL and completion state atomically from the client's perspective.
        filename = os.path.basename(final_video_path)
        output_parent = Path(final_video_path).resolve().parent
        if output_parent == Path(OUTPUT_VIDEO_DIR).resolve():
            result_url = f"/outputs/{filename}"
        elif output_parent == Path(OUTPUT_VIDEO_WAV2LIP_DIR).resolve():
            result_url = f"/outputs_fast/{filename}"
        else:
            raise RuntimeError(f"Pipeline returned an unexpected output path: {final_video_path}")

        jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "result_url": result_url,
            "error": None,
        })
        logger.info("Job %s completed: %s", job_id, final_video_path)
        publish_log("Output published and ready to download", level="success", source="server", job_id=job_id)

    except JobCancelledError:
        jobs[job_id].update({
            "status": "cancelled",
            "progress": 0,
            "result_url": None,
            "error": "Generation was ended by the user",
        })
        logger.info("Job %s cancelled", job_id)
        publish_log("Generation cancelled", level="warning", source="server", job_id=job_id)
    except Exception as e:
        jobs[job_id].update({"status": "failed", "error": str(e), "result_url": None})
        logger.exception("Job %s failed", job_id)
        publish_log(str(e), level="error", source="server", job_id=job_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
