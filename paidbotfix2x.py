"""
Telegram Highlight Clip Bot
===========================

A production-ready Telegram bot that:
- Accepts uploaded videos or YouTube links.
- Downloads YouTube videos via yt-dlp.
- Detects scene changes with PySceneDetect.
- Cuts highlight clips with FFmpeg (H.264 re-encode for reliability).
- Sends every generated clip back to the requesting user.
- Reports live progress, handles errors gracefully, and cleans up temp files.
- Persists per-user profile / history / settings in SQLite.
- Supports concurrent users with per-user cancellable jobs.

Single-file design as required. Run with: python bot.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite
from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.tl.types import PeerChannel
from telethon.errors import (
    ChannelPrivateError,
    ChannelInvalidError,
    UsernameNotOccupiedError,
    UsernameInvalidError,
    MessageIdInvalidError,
    FloodWaitError,
)
from dotenv import load_dotenv
from scenedetect import ContentDetector, SceneManager, open_video
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ---------------------------------------------------------------------------
# Configuration & logging
# ---------------------------------------------------------------------------

load_dotenv()

BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
API_ID = int(os.environ.get("API_ID", "0"))
API_HASH = os.environ.get("API_HASH", "").strip()
TELEGRAM_SESSION = os.environ.get("TELEGRAM_SESSION", "").strip()

# Telethon user-account client. Constructed here (no network yet); it is
# connected inside _post_init() so it shares python-telegram-bot's event loop.
tg_client = TelegramClient(
    StringSession(TELEGRAM_SESSION),
    API_ID,
    API_HASH,
)
OWNER_ID_RAW = os.environ.get("OWNER_ID", "").strip()
TEMP_FOLDER = Path(os.environ.get("TEMP_FOLDER", "temp")).resolve()
OUTPUT_FOLDER = Path(os.environ.get("OUTPUT_FOLDER", "clips")).resolve()
MAX_VIDEO_SIZE_MB = int(os.environ.get("MAX_VIDEO_SIZE_MB", "8192"))
DEFAULT_CLIP_LENGTH = int(os.environ.get("DEFAULT_CLIP_LENGTH", "15"))
SCENE_THRESHOLD = float(os.environ.get("SCENE_THRESHOLD", "22.0"))
MAX_CLIPS_PER_JOB = int(os.environ.get("MAX_CLIPS_PER_JOB", "25"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.environ.get("LOG_FILE", "bot.log")
DB_PATH = os.environ.get("DB_PATH", "bot_data.db")
STARTING_CREDITS = int(os.environ.get("STARTING_CREDITS", "3"))
CREDITS_PER_JOB = int(os.environ.get("CREDITS_PER_JOB", "1"))

# OWNER_ID accepts either a single ID or a comma-separated list (common typo
# is putting multiple IDs here instead of OWNER_IDS) — both are parsed the
# same way and merged into one OWNERS set. OWNER_ID keeps the *first* value
# for back-compat with any code/display that expects a single owner id.
OWNERS: set[int] = set()
for _part in (OWNER_ID_RAW + "," + os.environ.get("OWNER_IDS", "")).split(","):
    _part = _part.strip()
    if _part.lstrip("-").isdigit():
        OWNERS.add(int(_part))

OWNER_ID: Optional[int] = next(iter(sorted(OWNERS)), None)

TEMP_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
# yt-dlp is quite noisy at INFO; keep it a level down.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("yt_dlp").setLevel(logging.WARNING)
log = logging.getLogger("clipbot")

# ---------------------------------------------------------------------------
# Job registry (per-user, thread-safe via asyncio.Lock)
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """Represents an in-flight clip-generation job for a single user."""

    job_id: str
    user_id: int
    source: str  # "telegram", "youtube", or "telegram_link"
    source_label: str
    status: str = "queued"
    progress: str = ""
    started_at: float = field(default_factory=time.time)
    workdir: Optional[Path] = None
    task: Optional[asyncio.Task] = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    status_message_id: Optional[int] = None
    chat_id: Optional[int] = None
    clips_created: int = 0
    is_owner: bool = False
    credit_charged: bool = False


JOBS: dict[int, Job] = {}
JOBS_LOCK = asyncio.Lock()


# ---------------------------------------------------------------------------
# Persistence (SQLite via aiosqlite)
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id       INTEGER PRIMARY KEY,
    username      TEXT,
    first_name    TEXT,
    joined_at     TEXT NOT NULL,
    clip_length   INTEGER NOT NULL DEFAULT 15,
    threshold     REAL    NOT NULL DEFAULT 22.0,
    max_clips     INTEGER NOT NULL DEFAULT 25,
    credits       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL,
    source        TEXT    NOT NULL,
    source_label  TEXT,
    clips_created INTEGER NOT NULL DEFAULT 0,
    status        TEXT    NOT NULL,
    duration_sec  REAL    NOT NULL DEFAULT 0,
    created_at    TEXT    NOT NULL
);
"""


async def db_init() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(SCHEMA)
        await db.commit()
        # Migrate DBs created before the credits column existed.
        try:
            await db.execute("ALTER TABLE users ADD COLUMN credits INTEGER NOT NULL DEFAULT 0")
            await db.commit()
        except aiosqlite.OperationalError:
            pass  # column already present


async def db_upsert_user(user_id: int, username: Optional[str], first_name: Optional[str]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO users (user_id, username, first_name, joined_at, clip_length, threshold, max_clips, credits)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                    username=excluded.username,
                    first_name=excluded.first_name""",
            (user_id, username, first_name, now, DEFAULT_CLIP_LENGTH, SCENE_THRESHOLD,
             MAX_CLIPS_PER_JOB, STARTING_CREDITS),
        )
        await db.commit()


async def db_get_user(user_id: int) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
        row = await cur.fetchone()
        return dict(row) if row else None


async def db_update_setting(user_id: int, key: str, value) -> None:
    if key not in {"clip_length", "threshold", "max_clips"}:
        raise ValueError(f"Illegal setting key: {key}")
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(f"UPDATE users SET {key}=? WHERE user_id=?", (value, user_id))
        await db.commit()


async def db_get_credits(user_id: int) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT credits FROM users WHERE user_id=?", (user_id,))
        row = await cur.fetchone()
        return int(row[0]) if row else 0


async def db_add_credits(user_id: int, amount: int) -> int:
    """Add (amount > 0) or remove (amount < 0) credits; never drops below 0.
    Returns the resulting balance."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET credits = MAX(credits + ?, 0) WHERE user_id=?",
            (amount, user_id),
        )
        await db.commit()
        cur = await db.execute("SELECT credits FROM users WHERE user_id=?", (user_id,))
        row = await cur.fetchone()
        return int(row[0]) if row else 0


async def db_add_history(user_id: int, source: str, source_label: str,
                         clips_created: int, status: str, duration_sec: float) -> None:
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO history (user_id, source, source_label, clips_created, status, duration_sec, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, source, source_label, clips_created, status, duration_sec, now),
        )
        await db.commit()


async def db_get_history(user_id: int, limit: int = 10) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM history WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]


async def db_profile_stats(user_id: int) -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            """SELECT COUNT(*) AS jobs,
                      COALESCE(SUM(clips_created),0) AS clips,
                      COALESCE(SUM(duration_sec),0)  AS total_seconds
                 FROM history WHERE user_id=?""",
            (user_id,),
        )
        row = await cur.fetchone()
        return dict(row) if row else {"jobs": 0, "clips": 0, "total_seconds": 0}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
YOUTUBE_RE = re.compile(
    r"(https?://)?(www\.)?(youtube\.com/(watch\?v=|shorts/|embed/|v/)|youtu\.be/)[\w-]+",
    re.IGNORECASE,
)

TG_RE = re.compile(
    r"https://t\.me/(?:(c/\d+)|([A-Za-z0-9_]+))/(\d+)",
    re.IGNORECASE,
)

def is_youtube_url(text: str) -> bool:
    return bool(YOUTUBE_RE.search(text or ""))

def is_telegram_url(text: str) -> bool:
    return bool(TG_RE.search(text or ""))

def is_owner(user_id: int) -> bool:
    return user_id in OWNERS

def sanitize(name: str) -> str:
    """Return a filesystem-safe version of *name*."""
    name = re.sub(r"[^\w-.() ]+", "_", name).strip()
    return name[:80] or "video"

def fmt_bytes(n: int) -> str:
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < step:
            return f"{n:.1f} {unit}"
        n /= step
    return f"{n:.1f} PB"


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


async def edit_status(context: ContextTypes.DEFAULT_TYPE, job: Job, text: str) -> None:
    """Edit the persistent status message; ignore identical-content errors."""
    job.progress = text
    if job.chat_id is None or job.status_message_id is None:
        return
    try:
        await context.bot.edit_message_text(
            chat_id=job.chat_id,
            message_id=job.status_message_id,
            text=text,
            parse_mode=ParseMode.HTML,
        )
    except BadRequest as e:
        # Silently ignore "message is not modified"
        if "not modified" not in str(e).lower():
            log.debug("status edit failed: %s", e)
    except TelegramError as e:
        log.debug("status edit telegram error: %s", e)


# ---------------------------------------------------------------------------
# Video download & processing
# ---------------------------------------------------------------------------


async def run_ffmpeg(args: list[str]) -> tuple[int, str]:
    """Run ffmpeg asynchronously and return (returncode, stderr_text)."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", *args,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    return proc.returncode, stderr.decode("utf-8", errors="replace")


async def probe_duration(path: Path) -> float:
    """Return media duration in seconds using ffprobe. 0.0 on failure."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        return float(out.decode().strip())
    except (ValueError, FileNotFoundError):
        return 0.0


async def download_telegram_link(link: str, workdir: Path, job: Job,
                                 context: ContextTypes.DEFAULT_TYPE) -> Path:
    """
    Download the media of a Telegram post using the authenticated Telethon
    *user* client (not the Bot API), so 2GB–4GB+ files are supported.

    Handles:
      • Public channels/groups:   https://t.me/<username>/<msg_id>
      • Private channels/groups:  https://t.me/c/<internal_id>/<msg_id>

    Raises ValueError with a friendly message for invalid links, missing
    messages, messages without media, and inaccessible private chats.
    """
    if tg_client is None or not tg_client.is_connected():
        raise RuntimeError("Telethon client is not connected — cannot download Telegram links.")

    m = TG_RE.search(link or "")
    if not m:
        raise ValueError("That doesn't look like a valid Telegram post link.")

    c_part, username, msg_id_raw = m.group(1), m.group(2), m.group(3)
    msg_id = int(msg_id_raw)

    async def _fetch_message():
        if c_part:  # private chat form: c/<internal_id>
            channel_id = int(c_part.split("/", 1)[1])
            peer = PeerChannel(channel_id)
        else:
            peer = username
        return await tg_client.get_messages(peer, ids=msg_id)

    # ---- resolve + fetch the target message ----
    try:
        message = await _fetch_message()
    except (ChannelPrivateError,):
        raise ValueError(
            "🔒 That chat is private and the logged-in account can't access it. "
            "Make sure the account is a member of that channel/group."
        )
    except (UsernameNotOccupiedError, UsernameInvalidError):
        raise ValueError("That channel/username doesn't exist.")
    except (MessageIdInvalidError,):
        raise ValueError("That message doesn't exist or was deleted.")
    except (ValueError, ChannelInvalidError, TypeError):
        # Common for private (c/...) links whose entity isn't cached yet:
        # populate the dialog cache once, then retry.
        try:
            async for _ in tg_client.iter_dialogs():
                pass
            message = await _fetch_message()
        except Exception:
            raise ValueError(
                "Couldn't access that chat. Make sure the logged-in account "
                "is a member and the link is correct."
            )
    except FloodWaitError as e:
        raise ValueError(f"Telegram rate limit hit — try again in {e.seconds}s.")

    if message is None:
        raise ValueError("That message doesn't exist or was deleted.")
    if not getattr(message, "media", None):
        raise ValueError("That Telegram message doesn't contain any downloadable media.")

    # ---- throttled progress updates ----
    last_update = 0.0

    async def _progress(received: int, total: int) -> None:
        nonlocal last_update
        if job.cancel_event.is_set():
            raise asyncio.CancelledError()
        now = time.time()
        if now - last_update < 2.0:
            return
        last_update = now
        pct = (received / total * 100) if total else 0
        await edit_status(
            context, job,
            f"📥 <b>Downloading Telegram media…</b>\n"
            f"{fmt_bytes(received)} / {fmt_bytes(total) if total else '?'} ({pct:.1f}%)",
        )

    dest = workdir / "source"
    out = await tg_client.download_media(
        message, file=str(dest), progress_callback=_progress
    )
    if not out:
        raise ValueError("Failed to download the media from that message.")
    path = Path(out)
    if not path.exists():
        raise FileNotFoundError("Telegram download did not produce a file.")
    return path


async def download_youtube(url: str, workdir: Path, job: Job,
                           context: ContextTypes.DEFAULT_TYPE) -> Path:
    """Download a YouTube video with yt-dlp; returns the resulting file path."""
    import yt_dlp  # imported lazily to keep startup light

    loop = asyncio.get_running_loop()
    last_update = 0.0

    def progress_hook(d: dict) -> None:
        nonlocal last_update
        if job.cancel_event.is_set():
            raise yt_dlp.utils.DownloadError("Cancelled by user")
        if d.get("status") == "downloading":
            now = time.time()
            if now - last_update < 2.0:
                return
            last_update = now
            downloaded = d.get("downloaded_bytes") or 0
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            pct = (downloaded / total * 100) if total else 0
            msg = (f"📥 <b>Downloading YouTube video…</b>\n"
                   f"{fmt_bytes(downloaded)} / {fmt_bytes(total) if total else '?'} "
                   f"({pct:.1f}%)")
            asyncio.run_coroutine_threadsafe(edit_status(context, job, msg), loop)

    outtmpl = str(workdir / "source.%(ext)s")
    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [progress_hook],
        "concurrent_fragment_downloads": 4,
        "max_filesize": MAX_VIDEO_SIZE_MB * 1024 * 1024,
        "retries": 3,
    }

    def _download() -> str:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)

    filename = await asyncio.to_thread(_download)
    path = Path(filename)
    # yt-dlp may have written a different container after merge
    if not path.exists():
        # look for merged file with same stem
        for candidate in workdir.glob("source.*"):
            path = candidate
            break
    if not path.exists():
        raise FileNotFoundError("YouTube download did not produce a file")
    return path


def detect_scenes_sync(video_path: Path, threshold: float) -> list[tuple[float, float]]:
    """Blocking scene detection. Returns list of (start_sec, end_sec)."""
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=False)
    scenes = scene_manager.get_scene_list()
    return [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]


async def detect_scenes(video_path: Path, threshold: float,
                        cancel_event: asyncio.Event) -> list[tuple[float, float]]:
    """Run scene detection in a worker thread; honour cancellation on completion."""
    scenes = await asyncio.to_thread(detect_scenes_sync, video_path, threshold)
    if cancel_event.is_set():
        raise asyncio.CancelledError()
    return scenes


async def cut_clip(source: Path, out_path: Path, start: float, end: float) -> None:
    """Cut a clip using ffmpeg with H.264 re-encoding (reliable, precise cuts)."""
    duration = max(0.1, end - start)
    args = [
        "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", str(source),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    rc, err = await run_ffmpeg(args)
    if rc != 0:
        raise RuntimeError(f"ffmpeg cut failed: {err.strip()[:300]}")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


async def process_video_job(update: Update, context: ContextTypes.DEFAULT_TYPE,
                            job: Job, source_kind: str, source_ref) -> None:
    """
    Full pipeline:
      1. Acquire source (Telegram file / YouTube URL)
      2. Scene-detect
      3. Cut clips
      4. Upload clips
      5. Cleanup
    """
    user = update.effective_user
    user_settings = await db_get_user(user.id) or {}
    threshold = float(user_settings.get("threshold") or SCENE_THRESHOLD)
    min_clip = int(user_settings.get("clip_length") or DEFAULT_CLIP_LENGTH)
    max_clips = int(user_settings.get("max_clips") or MAX_CLIPS_PER_JOB)

    workdir = TEMP_FOLDER / job.job_id
    workdir.mkdir(parents=True, exist_ok=True)
    job.workdir = workdir
    clips_dir = OUTPUT_FOLDER / job.job_id
    clips_dir.mkdir(parents=True, exist_ok=True)

    final_status = "failed"
    try:
        # ---------- 1. Acquire source ----------
        if source_kind == "youtube":
            await edit_status(context, job, "📥 <b>Fetching YouTube video…</b>")
            source_path = await download_youtube(source_ref, workdir, job, context)

        elif source_kind == "telegram_link":
            await edit_status(context, job, "📥 <b>Downloading from Telegram link…</b>")
            source_path = await download_telegram_link(source_ref, workdir, job, context)

        else:  # Telegram bot upload
            await edit_status(context, job, "📥 <b>Downloading your video…</b>")
            tg_file = await context.bot.get_file(source_ref)
            source_path = workdir / "source.mp4"
            await tg_file.download_to_drive(custom_path=str(source_path))

        if job.cancel_event.is_set():
            raise asyncio.CancelledError()

        size_mb = source_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_VIDEO_SIZE_MB:
            raise ValueError(f"Video too large ({size_mb:.1f} MB > {MAX_VIDEO_SIZE_MB} MB)")

        duration = await probe_duration(source_path)
        job.status = "detecting"
        await edit_status(
            context, job,
            f"🔎 <b>Detecting scenes…</b>\n"
            f"Video: {fmt_duration(duration)} · {fmt_bytes(source_path.stat().st_size)}\n"
            f"Threshold: {threshold}",
        )

        # ---------- 2. Scene detection ----------
        scenes = await detect_scenes(source_path, threshold, job.cancel_event)
        if not scenes:
            # Fallback: split into fixed windows of `min_clip` seconds
            step = max(min_clip, 5)
            scenes = [(t, min(t + step, duration)) for t in range(0, int(duration), step)]

        # Filter tiny scenes and cap
        scenes = [s for s in scenes if (s[1] - s[0]) >= 1.0][:max_clips]
        if not scenes:
            raise RuntimeError("No usable scenes were detected in this video.")

        # ---------- 3. Cut clips ----------
        job.status = "cutting"
        total = len(scenes)
        cut_paths: list[Path] = []
        for idx, (start, end) in enumerate(scenes, 1):
            if job.cancel_event.is_set():
                raise asyncio.CancelledError()
            await edit_status(
                context, job,
                f"✂️ <b>Cutting clip {idx}/{total}</b>\n"
                f"Segment: {fmt_duration(start)} → {fmt_duration(end)} "
                f"({end - start:.1f}s)",
            )
            out = clips_dir / f"clip_{idx:02d}.mp4"
            try:
                await cut_clip(source_path, out, start, end)
                cut_paths.append(out)
            except Exception as e:
                log.warning("clip %d failed: %s", idx, e)

        if not cut_paths:
            raise RuntimeError("All clips failed to encode. Check FFmpeg installation.")

        # ---------- 4. Upload ----------
        job.status = "uploading"
        for idx, path in enumerate(cut_paths, 1):
            if job.cancel_event.is_set():
                raise asyncio.CancelledError()
            await edit_status(
                context, job,
                f"📤 <b>Sending clip {idx}/{len(cut_paths)}…</b>",
            )
            try:
                await context.bot.send_chat_action(job.chat_id, ChatAction.UPLOAD_VIDEO)
                with path.open("rb") as fh:
                    await context.bot.send_video(
                        chat_id=job.chat_id,
                        video=fh,
                        caption=f"🎬 Highlight {idx}/{len(cut_paths)}",
                        supports_streaming=True,
                        read_timeout=120,
                        write_timeout=600,
                        connect_timeout=60,
                    )
                job.clips_created += 1
            except TelegramError as e:
                log.warning("upload of %s failed: %s", path.name, e)
                await context.bot.send_message(
                    job.chat_id,
                    f"⚠️ Couldn't send clip {idx}: {e}",
                )

        final_status = "done"
        await edit_status(
            context, job,
            f"✅ <b>Done!</b> Sent {job.clips_created} clip(s).",
        )

    except asyncio.CancelledError:
        final_status = "cancelled"
        await edit_status(context, job, "🛑 <b>Job cancelled.</b>")
        raise
    except Exception as e:
        final_status = "failed"
        log.exception("job %s failed", job.job_id)
        await edit_status(
            context, job,
            f"❌ <b>Job failed:</b> <code>{str(e)[:300]}</code>",
        )
    finally:
        # Refund the reserved credit if the job didn't actually complete.
        if job.credit_charged and not job.is_owner and final_status != "done":
            try:
                await db_add_credits(job.user_id, CREDITS_PER_JOB)
            except Exception as e:
                log.warning("credit refund failed for %s: %s", job.user_id, e)

        elapsed = time.time() - job.started_at
        try:
            await db_add_history(
                user_id=job.user_id,
                source=job.source,
                source_label=job.source_label,
                clips_created=job.clips_created,
                status=final_status,
                duration_sec=elapsed,
            )
        except Exception as e:
            log.warning("history write failed: %s", e)

        # Cleanup temp + output folders
        for folder in (workdir, clips_dir):
            try:
                if folder.exists():
                    shutil.rmtree(folder, ignore_errors=True)
            except Exception as e:
                log.warning("cleanup failed for %s: %s", folder, e)

        async with JOBS_LOCK:
            JOBS.pop(job.user_id, None)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    u = update.effective_user
    await db_upsert_user(u.id, u.username, u.first_name)
    if is_owner(u.id):
        credit_line = "You're an <b>owner</b> — unlimited credits."
    else:
        credits = await db_get_credits(u.id)
        credit_line = f"You have <b>{credits}</b> credit(s) (1 job = {CREDITS_PER_JOB} credit)."
    text = (
        f"👋 <b>Hello {u.first_name or 'there'}!</b>\n\n"
        "I'm a <b>Highlight Clip Bot</b>. Send me a video, a YouTube link, or a "
        "Telegram post link and I'll auto-detect scene changes and cut highlight "
        "clips for you.\n\n"
        f"{credit_line}\n\n"
        "Use /help to see what I can do."
    )
    await update.effective_message.reply_html(text)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "<b>📖 Help</b>\n\n"
        "<b>How to use:</b>\n"
        "• Send a video file (up to "
        f"{MAX_VIDEO_SIZE_MB} MB) — I'll process it automatically.\n"
        "• Or send a YouTube link and I'll download + process it.\n"
        "• Or paste a Telegram post link (public/private channels, groups, "
        "supergroups) — I'll fetch the media via a user account (2GB–4GB+ supported).\n"
        "• Use /clip &lt;url&gt; to explicitly submit a link.\n\n"
        "<b>Commands:</b>\n"
        "/start — greet\n"
        "/help — this message\n"
        "/clip &lt;url&gt; — submit a YouTube link\n"
        "/status — current job status\n"
        "/settings — view/change your preferences\n"
        "/profile — your stats\n"
        "/history — last 10 jobs\n"
        "/cancel — cancel the running job\n"
        "/about — about this bot\n"
        f"/addcredits &lt;user_id&gt; &lt;amount&gt; — owner-only, grant/remove credits\n\n"
        f"<b>Credits:</b> each job costs {CREDITS_PER_JOB}. Owners have unlimited credits."
    )
    await update.effective_message.reply_html(text)


async def cmd_about(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "<b>🎬 Highlight Clip Bot</b>\n"
        "Python 3.12 · python-telegram-bot · yt-dlp · PySceneDetect · FFmpeg\n\n"
        "Detects scene changes and produces MP4 highlight clips (H.264 + AAC).\n"
        f"Owners: <code>{', '.join(str(o) for o in sorted(OWNERS)) or 'not configured'}</code>"
    )
    await update.effective_message.reply_html(text)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    async with JOBS_LOCK:
        job = JOBS.get(uid)
    if not job:
        await update.effective_message.reply_text("😴 No active job. Send a video or link to start.")
        return
    elapsed = time.time() - job.started_at
    text = (
        f"<b>📊 Status</b>\n"
        f"Job: <code>{job.job_id}</code>\n"
        f"Source: {job.source} — {job.source_label}\n"
        f"State: <b>{job.status}</b>\n"
        f"Elapsed: {fmt_duration(elapsed)}\n"
        f"Clips sent: {job.clips_created}\n"
        f"{job.progress or ''}"
    )
    await update.effective_message.reply_html(text)


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    async with JOBS_LOCK:
        job = JOBS.get(uid)
    if not job:
        await update.effective_message.reply_text("Nothing to cancel — no active job.")
        return
    job.cancel_event.set()
    if job.task and not job.task.done():
        job.task.cancel()
    await update.effective_message.reply_text("🛑 Cancelling your job…")


async def cmd_profile(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    u = update.effective_user
    await db_upsert_user(u.id, u.username, u.first_name)
    user = await db_get_user(u.id)
    stats = await db_profile_stats(u.id)
    joined = (user or {}).get("joined_at", "")
    try:
        joined_disp = datetime.fromisoformat(joined).strftime("%Y-%m-%d")
    except Exception:
        joined_disp = joined or "unknown"
    credits_disp = "♾️ Unlimited (owner)" if is_owner(u.id) else str((user or {}).get("credits", 0))
    text = (
        f"<b>👤 Profile</b>\n"
        f"Name: {u.first_name}\n"
        f"Username: @{u.username or '—'}\n"
        f"User ID: <code>{u.id}</code>\n"
        f"Joined: {joined_disp}\n"
        f"Credits: <b>{credits_disp}</b>\n\n"
        f"<b>📈 Stats</b>\n"
        f"Total jobs: {stats['jobs']}\n"
        f"Total clips produced: {stats['clips']}\n"
        f"Total processing time: {fmt_duration(stats['total_seconds'])}"
    )
    await update.effective_message.reply_html(text)


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rows = await db_get_history(update.effective_user.id, limit=10)
    if not rows:
        await update.effective_message.reply_text("🗒️ History is empty. Send a video to get started!")
        return
    lines = ["<b>🗒️ Recent jobs</b>"]
    for r in rows:
        try:
            dt = datetime.fromisoformat(r["created_at"]).strftime("%Y-%m-%d %H:%M")
        except Exception:
            dt = r["created_at"]
        emoji = {"done": "✅", "failed": "❌", "cancelled": "🛑"}.get(r["status"], "•")
        label = (r["source_label"] or "")[:40]
        lines.append(
            f"{emoji} <b>{dt}</b> · {r['source']} · {label}\n"
            f"   clips: {r['clips_created']} · {fmt_duration(r['duration_sec'])}"
        )
    await update.effective_message.reply_html("\n".join(lines))


async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    u = update.effective_user
    await db_upsert_user(u.id, u.username, u.first_name)
    user = await db_get_user(u.id) or {}

    args = context.args or []
    if not args:
        text = (
            "<b>⚙️ Your settings</b>\n"
            f"• <b>clip_length</b> (min clip seconds): <code>{user.get('clip_length', DEFAULT_CLIP_LENGTH)}</code>\n"
            f"• <b>threshold</b> (scene sensitivity): <code>{user.get('threshold', SCENE_THRESHOLD)}</code>\n"
            f"• <b>max_clips</b> (per job): <code>{user.get('max_clips', MAX_CLIPS_PER_JOB)}</code>\n\n"
            "<b>Change:</b>\n"
            "<code>/settings clip_length 20</code>\n"
            "<code>/settings threshold 18.0</code>\n"
            "<code>/settings max_clips 15</code>\n\n"
            "Lower <b>threshold</b> = more cuts (more sensitive)."
        )
        await update.effective_message.reply_html(text)
        return

    if len(args) != 2:
        await update.effective_message.reply_text("Usage: /settings <key> <value>")
        return

    key, raw = args[0].lower(), args[1]
    try:
        if key == "clip_length":
            v = int(raw)
            if not (1 <= v <= 600):
                raise ValueError("clip_length must be 1-600")
            await db_update_setting(u.id, "clip_length", v)
        elif key == "threshold":
            v = float(raw)
            if not (5.0 <= v <= 100.0):
                raise ValueError("threshold must be 5.0-100.0")
            await db_update_setting(u.id, "threshold", v)
        elif key == "max_clips":
            v = int(raw)
            if not (1 <= v <= 100):
                raise ValueError("max_clips must be 1-100")
            await db_update_setting(u.id, "max_clips", v)
        else:
            raise ValueError(f"Unknown setting '{key}'")
    except ValueError as e:
        await update.effective_message.reply_text(f"❌ {e}")
        return

    await update.effective_message.reply_text(f"✅ Updated {key} → {raw}")


async def cmd_addcredits(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Owner-only: /addcredits <user_id> <amount>  (amount may be negative to deduct)."""
    u = update.effective_user
    if not is_owner(u.id):
        await update.effective_message.reply_text("⛔ This command is owner-only.")
        return

    args = context.args or []
    if len(args) != 2:
        await update.effective_message.reply_text("Usage: /addcredits <user_id> <amount>")
        return

    try:
        target_id = int(args[0])
        amount = int(args[1])
    except ValueError:
        await update.effective_message.reply_text("❌ <user_id> and <amount> must both be integers.")
        return

    # Make sure the target user has a row before we touch their balance.
    if not await db_get_user(target_id):
        await db_upsert_user(target_id, None, None)

    new_balance = await db_add_credits(target_id, amount)
    verb = "Added" if amount >= 0 else "Removed"
    prep = "to" if amount >= 0 else "from"
    await update.effective_message.reply_html(
        f"✅ {verb} <code>{abs(amount)}</code> credit(s) {prep} <code>{target_id}</code>.\n"
        f"New balance: <code>{new_balance}</code>"
        + (" (owner — unlimited regardless of balance)" if is_owner(target_id) else "")
    )


async def cmd_clip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.effective_message.reply_text(
            "Usage: /clip <YouTube or Telegram post URL>\nOr just paste the URL directly."
        )
        return
    url = context.args[0]
    if is_youtube_url(url):
        await _kick_off_job(update, context, source_kind="youtube", source_ref=url,
                           label=url)
    elif is_telegram_url(url):
        await _kick_off_job(update, context, source_kind="telegram_link", source_ref=url,
                           label=url)
    else:
        await update.effective_message.reply_text(
            "❌ That doesn't look like a YouTube or Telegram post URL."
        )


# ---------------------------------------------------------------------------
# Message handlers (video uploads, YouTube URLs)
# ---------------------------------------------------------------------------


async def on_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    video = msg.video or msg.document
    if not video:
        return
    if getattr(video, "file_size", None) and video.file_size > MAX_VIDEO_SIZE_MB * 1024 * 1024:
        await msg.reply_text(
            f"❌ File too large ({fmt_bytes(video.file_size)}). "
            f"Max allowed: {MAX_VIDEO_SIZE_MB} MB."
        )
        return
    label = getattr(video, "file_name", None) or "telegram_video.mp4"
    await _kick_off_job(update, context, source_kind="telegram",
                        source_ref=video.file_id, label=label)


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.effective_message.text or "").strip()

    if is_youtube_url(text):
        await _kick_off_job(
            update,
            context,
            source_kind="youtube",
            source_ref=text,
            label=text,
        )
        return

    if is_telegram_url(text):
        await _kick_off_job(
            update,
            context,
            source_kind="telegram_link",
            source_ref=text,
            label=text,
        )
        return

    await update.effective_message.reply_text(
        "🤔 Send me a video, a YouTube link, or a Telegram post link, or use /help to see commands."
    )


async def _kick_off_job(update: Update, context: ContextTypes.DEFAULT_TYPE,
                        source_kind: str, source_ref, label: str) -> None:
    u = update.effective_user
    await db_upsert_user(u.id, u.username, u.first_name)

    owner = is_owner(u.id)
    if not owner:
        credits = await db_get_credits(u.id)
        if credits < CREDITS_PER_JOB:
            await update.effective_message.reply_html(
                "🚫 <b>Out of credits.</b>\n"
                f"You have <code>{credits}</code>, this needs <code>{CREDITS_PER_JOB}</code>.\n"
                "Contact the bot owner to top up."
            )
            return

    async with JOBS_LOCK:
        if u.id in JOBS:
            await update.effective_message.reply_text(
                "⏳ You already have a job in progress. Use /status or /cancel."
            )
            return
        job = Job(
            job_id=uuid.uuid4().hex[:10],
            user_id=u.id,
            source=source_kind,
            source_label=sanitize(label),
            is_owner=owner,
        )
        JOBS[u.id] = job

    if not owner:
        # Reserve the credit now; refunded automatically if the job fails/cancels.
        await db_add_credits(u.id, -CREDITS_PER_JOB)
        job.credit_charged = True

    msg = await update.effective_message.reply_html(
        f"⏳ <b>Queued</b> — job <code>{job.job_id}</code>"
    )
    job.chat_id = msg.chat_id
    job.status_message_id = msg.message_id

    async def _runner() -> None:
        try:
            await process_video_job(update, context, job, source_kind, source_ref)
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("runner crashed")

    job.task = asyncio.create_task(_runner())


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.exception("Unhandled error: %s", context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "⚠️ Something went wrong. Please try again."
            )
        except TelegramError:
            pass


# ---------------------------------------------------------------------------
# Bootstrap / entry point
# ---------------------------------------------------------------------------


def check_binaries() -> None:
    """Fail fast if ffmpeg/ffprobe are missing."""
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            log.error("Required binary '%s' not found on PATH. "
                      "Install FFmpeg — see README.", tool)
            sys.exit(2)


async def _post_init(app: Application) -> None:
    await db_init()
    # Connect the Telethon user client inside PTB's event loop so both share
    # the same loop. The StringSession is already authenticated, so connect()
    # is enough — no phone/code prompt.
    try:
        await tg_client.connect()
        if await tg_client.is_user_authorized():
            me = await tg_client.get_me()
            uname = getattr(me, "username", None) or getattr(me, "id", "?")
            log.info("Telethon client connected & authorized as @%s.", uname)
        else:
            log.error(
                "Telethon session is NOT authorized. Telegram post links will "
                "not work until a valid TELEGRAM_SESSION is provided."
            )
    except Exception as e:
        log.error("Telethon connect failed: %s", e)
    log.info("Bot ready. Temp=%s  Output=%s", TEMP_FOLDER, OUTPUT_FOLDER)


async def _post_shutdown(app: Application) -> None:
    try:
        if tg_client.is_connected():
            await tg_client.disconnect()
            log.info("Telethon client disconnected.")
    except Exception as e:
        log.debug("Telethon disconnect error: %s", e)


def main() -> None:
    if not BOT_TOKEN or BOT_TOKEN.startswith("YOUR_"):
        log.error("BOT_TOKEN missing in .env — copy .env.example to .env and set it.")
        sys.exit(1)
    check_binaries()

    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .concurrent_updates(True)
        .build()
    )

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("about", cmd_about))
    app.add_handler(CommandHandler("clip", cmd_clip))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CommandHandler("profile", cmd_profile))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("settings", cmd_settings))
    app.add_handler(CommandHandler("addcredits", cmd_addcredits))

    # Media & text
    app.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO, on_video))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    app.add_error_handler(on_error)

    log.info("Starting bot (polling)…")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        stop_signals=(signal.SIGINT, signal.SIGTERM),
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
