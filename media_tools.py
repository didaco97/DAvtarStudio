import os
import shutil


def get_ffmpeg_executable() -> str:
    """Return a usable ffmpeg executable without requiring a system install."""
    configured = os.environ.get("FFMPEG_BINARY")
    if configured:
        configured = os.path.abspath(os.path.expanduser(configured))
        if os.path.isfile(configured):
            return configured
        raise RuntimeError(f"FFMPEG_BINARY does not point to a file: {configured}")

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg

        bundled_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "ffmpeg is unavailable. Install project dependencies with "
            "'python -m pip install -r requirements.txt' or set FFMPEG_BINARY."
        ) from exc

    if not os.path.isfile(bundled_ffmpeg):
        raise RuntimeError(f"Bundled ffmpeg executable was not found: {bundled_ffmpeg}")
    return bundled_ffmpeg


def require_nonempty_file(file_path: str, description: str) -> str:
    if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
        raise RuntimeError(f"{description} was not created: {file_path}")
    return file_path
