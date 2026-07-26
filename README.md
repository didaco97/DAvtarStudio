# DAvtar Studio

DAvtar creates lip-synced avatar videos from a face video and an audio track. It provides a FastAPI web UI and a command-line workflow.

The standard path produces a draft MP4. The HD path extracts those frames, enhances them with Real-ESRGAN/GFPGAN, and stitches them into an H.264/AAC MP4.

## Requirements

### Hardware and software

- Windows 10/11 or an equivalent Linux environment.
- Python 3.10–3.12. Python 3.10/3.11 is recommended for the legacy `librosa` and `numba` pins.
- NVIDIA GPU with compatible PyTorch/CUDA for practical processing times. CPU mode is supported but slower.
- Several GB of disk space for checkpoints, model weights, frames, and generated videos.
- Git, PowerShell, and internet access for the first model downloads.

The exact Python packages are listed in [requirements.txt](requirements.txt). `imageio-ffmpeg==0.6.0` supplies a project-local ffmpeg binary; a system ffmpeg installation is optional.

The application has no authentication and binds to `0.0.0.0:8000`. Keep it on a trusted machine or add authentication/restrict the bind address before exposing it to a network.

## Installation

Clone the repository and open PowerShell in the cloned project:

    git clone https://github.com/didaco97/DAvtarStudio.git
    cd DAvtarStudio

Create and activate a virtual environment:

    py -3.10 -m venv venv
    .\venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip setuptools wheel

Install the root dependencies and Google Drive downloader:

    python -m pip install -r requirements.txt
    python -m pip install gdown

Install Real-ESRGAN in editable mode:

    python -m pip install -r Real-ESRGAN\requirements.txt
    python Real-ESRGAN\setup.py develop

The bundled [setup.ps1](setup.ps1) performs the same broad sequence:

    Set-ExecutionPolicy -Scope Process Bypass
    .\setup.ps1

If activation is blocked, call the virtual-environment interpreter explicitly:

    .\venv\Scripts\python.exe -m pip install -r requirements.txt

## Checkpoints and model weights

The following files are expected in `checkpoints/`:

| File | Purpose |
|---|---|
| Generator checkpoint (`*.pth`) | Core animation model |
| `face_segmentation.pth` | Optional segmentation model |
| `esrgan_yunying.pth` | Optional draft super-resolution model |
| `pretrained.state` | Supporting project state |

Download the project checkpoints and sample video with:

    python download_models.py

That script uses the Google Drive IDs embedded in the script. If it cannot access Google Drive, download the upstream model checkpoints manually and place them at the paths above.

The HD path also needs the Real-ESRGAN `RealESRGAN_x4plus` and GFPGAN weights. They may download automatically on the first HD job into `Real-ESRGAN/weights/` and the installed GFPGAN weights directory. Run one HD job while online before using the application offline.

## Start the web application

Run one worker; the pipeline shares GPU resources and intermediate files and is serialized with an in-process lock:

    .\venv\Scripts\python.exe app.py

Open `http://localhost:8000`.

The UI accepts:

- Video: `.mp4`, `.m4v`, `.mov`, `.avi`, `.webm`, `.mkv`.
- Audio: `.wav`, `.mp3`, `.m4a`, `.aac`, `.ogg`, `.flac`.

Disable **HD Enhancement** for the faster draft path. Enable it for Real-ESRGAN/GFPGAN enhancement. The UI includes a generation timer and a live backend terminal with pause, follow-tail, and clear controls.

Job state is process-local. Restarting the server loses old `/api/status` records even if the output files remain. Do not use multiple Uvicorn workers with the current implementation.

## Command-line workflow

### Standard draft output

From the repository root:

    .\venv\Scripts\python.exe inference.py --checkpoint_path checkpoints\<generator-checkpoint>.pth --segmentation_path checkpoints\face_segmentation.pth --sr_path checkpoints\esrgan_yunying.pth --face input_videos\mona.mp4 --audio input_audios\ai.wav --no_sr --no_segmentation --outfile <draft-output>.mp4

Replace `<generator-checkpoint>.pth` and `<draft-output>.mp4` with paths in your checkout. The segmentation and super-resolution paths are required command-line arguments even when those features are disabled.

### Manual HD stages

    .\venv\Scripts\python.exe video2frames.py --input_video <draft-output>.mp4 --frames_path <draft-frames>\mona
    Push-Location Real-ESRGAN
    ..\venv\Scripts\python.exe inference_realesrgan.py -n RealESRGAN_x4plus -i <draft-frames>\mona --output ..\frames_hd\mona --outscale 3.5 --face_enhance
    Pop-Location

The web processor performs the final ffmpeg stitch using `frame_%05d_out.jpg`, the source frame rate, H.264/AAC, and `+faststart` metadata. [run_final.sh](run_final.sh) is retained as the original Linux-style reference.

## HTTP API

The API runs on port `8000`.

### `GET /`

Returns the web interface.

### `POST /api/generate`

Starts a background job using `multipart/form-data`:

| Field | Type | Required | Description |
|---|---|---:|---|
| `video` | file | yes | Supported video |
| `audio` | file | yes | Supported audio |
| `use_esrgan` | boolean | no | Defaults to `true` |

Success response:

    {"job_id":"b9167792-..."}

PowerShell example:

    $job = curl.exe -sS -X POST -F "video=@input_videos\mona.mp4;type=video/mp4" -F "audio=@input_audios\ai.wav;type=audio/wav" -F "use_esrgan=false" http://localhost:8000/api/generate | ConvertFrom-Json
    $job.job_id

Unsupported extensions and empty uploads return `400`.

### `GET /api/status/{job_id}`

Returns:

    {
      "status": "processing",
      "progress": 0,
      "result_url": null,
      "error": null
    }

Statuses are `processing`, `completed`, and `failed`. Completed jobs return `/outputs/{job_id}.mp4` for HD or `/outputs_fast/{job_id}.mp4` for draft. Unknown jobs return `404`.

Polling example:

    do {
      Start-Sleep -Seconds 3
      $status = curl.exe -sS "http://localhost:8000/api/status/$($job.job_id)" | ConvertFrom-Json
      $status | Format-List
    } while ($status.status -eq 'processing')

### `GET /api/logs/recent?limit=250`

Returns up to 1,000 recent structured events. Each entry contains `id`, `timestamp`, `level`, `source`, `job_id`, and `message`.

### `GET /api/logs/stream`

Opens the Server-Sent Events stream used by the frontend terminal. Events have this shape:

    id: 148
    event: log
    data: {"level":"success","source":"server","job_id":"b916...","message":"Output published and ready to download"}

The endpoint sends keep-alive comments and honors `Last-Event-ID` on reconnect.

### Static and output routes

| Route | Directory/content |
|---|---|
| `/static/...` | Frontend assets |
| `/outputs/{filename}` | `output_videos_hd/` |
| `/outputs_fast/{filename}` | Configured draft-output directory |

The output mounts support HTTP range requests for browser streaming and seeking.

## Testing

Run the regression suite:

    .\venv\Scripts\python.exe -m unittest discover -s tests -v

Run syntax checks:

    .\venv\Scripts\python.exe -m py_compile app.py processor.py inference.py runtime_logs.py
    node --check static\app.js

Check the server and recent logs:

    curl.exe -i http://localhost:8000/
    curl.exe -sS http://localhost:8000/api/logs/recent?limit=20

Inspect the SSE stream for five seconds:

    curl.exe --max-time 5 -N http://localhost:8000/api/logs/stream

For a complete manual test, upload a short video/audio pair, poll the returned job until it is terminal, then request its `result_url` and confirm HTTP `200` plus browser playback.

## Output directories

| Directory | Purpose |
|---|---|
| `input_videos/` | Uploaded and CLI videos |
| `input_audios/` | Uploaded and CLI audio |
| Draft-output directory | Draft MP4 files |
| `output_videos_hd/` | Enhanced MP4 files |
| Draft-frame directory | Frames before enhancement |
| `frames_hd/` | Enhanced frames before stitching |
| `temp/` | Intermediate AVI/audio files |
| `checkpoints/` | Animation-model checkpoints |
| `Real-ESRGAN/weights/` | Real-ESRGAN weights |

Uploads and frame directories are not automatically purged. Remove old artifacts periodically.

## Troubleshooting

### ffmpeg is not found

Reinstall the root dependencies:

    .\venv\Scripts\python.exe -m pip install -r requirements.txt

The application detects the project-local `imageio-ffmpeg` binary and a system/WinGet ffmpeg binary on `PATH`.

### A job fails

Inspect the Backend runtime terminal or `/api/logs/recent`. The final error identifies the failed stage. Check that inputs are non-empty and all animation-model checkpoints exist.

### The first HD job fails or is very slow

Real-ESRGAN/GFPGAN may be downloading weights. Keep the first HD job online and inspect the weights directories afterward.

### CUDA out of memory

Use draft mode, close other GPU applications, or lower the batch size in `processor.py` from its current web value of `16`. CPU mode is slower but avoids GPU memory pressure.

### Browser preview fails after completion

Use the Download Result link and inspect the MP4 with ffmpeg/ffprobe. Outputs are written as H.264/AAC MP4 with `+faststart`; the frontend retries preview loading and preserves the download link if decoding still fails.

### Job status disappears after restart

Jobs are stored in an in-memory dictionary. Persistent storage is required for multi-worker or production deployment.

### PyTorch `torch.load` warning

The `weights_only=False` message is a security warning, not a generation error. Only load trusted checkpoint files. `weights_only=True` can be enabled after confirming every project checkpoint contains compatible tensor/state-dict data.

## Acknowledgements

This project combines upstream lip-sync, Real-ESRGAN, GFPGAN, BasicSR, face-parsing, and face-detection projects. Review their original licenses and model terms before redistribution or commercial use.
