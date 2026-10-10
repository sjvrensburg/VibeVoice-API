from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerConfig:
    host: str = os.environ.get("VIBEVOICE_API_HOST", "0.0.0.0")
    port: int = int(os.environ.get("VIBEVOICE_API_PORT", "8000"))
    base_path: str = os.environ.get("VIBEVOICE_API_BASE_PATH", "/v1")
    # Path or HF ID of the model to load by default
    model_path: str = os.environ.get("VIBEVOICE_MODEL", "microsoft/VibeVoice-1.5B")
    # Device preference: auto, cuda, mps, cpu
    device_preference: str = os.environ.get("VIBEVOICE_DEVICE", "auto")
    # Number of diffusion steps for inference
    ddpm_steps: int = int(os.environ.get("VIBEVOICE_DDPM_STEPS", "10"))
    # Default audio sample rate returned by the model
    sample_rate: int = int(os.environ.get("VIBEVOICE_SAMPLE_RATE", "24000"))
    # Optional: path to ffmpeg binary for extra encodings (mp3/flac/opus/aac)
    ffmpeg_path: str = os.environ.get("VIBEVOICE_FFMPEG", "ffmpeg")
    # Validation toggle for local shards (1=on, 0=off)
    validate_shards: bool = os.environ.get("VIBEVOICE_VALIDATE_SHARDS", "1") not in {"0", "false", "False"}
    # ---- Synthesis verification (off by default) --------------------------
    # VibeVoice occasionally produces a garbled clip: fluent-sounding audio that
    # is not the requested text. Measured on a real 59-clip podcast episode:
    # 1 clip (1.7%) came out as grammatical nonsense ("These are illiterary
    # enterprises"). It is stochastic, not text-triggered -- the exact text that
    # failed in production synthesized cleanly 20/20 times on re-test -- so it
    # cannot be avoided by prompt or model choice, only detected.
    #
    # When enabled, each clip is transcribed with whisper.cpp and compared to
    # the input text; below the threshold it is re-synthesized. Clean clips
    # score ~1.000 and garbled ones score far below, so the separation is wide
    # and the threshold is not delicate.
    #
    # Off by default: it adds an ASR pass per clip and a hard dependency on a
    # whisper.cpp binary + model that most deployments will not have.
    verify: bool = os.environ.get("VIBEVOICE_VERIFY", "0") not in {"0", "false", "False", ""}
    verify_threshold: float = float(os.environ.get("VIBEVOICE_VERIFY_THRESHOLD", "0.75"))
    verify_retries: int = int(os.environ.get("VIBEVOICE_VERIFY_RETRIES", "2"))
    verify_whisper_bin: str = os.environ.get("VIBEVOICE_VERIFY_WHISPER_BIN", "")
    verify_whisper_model: str = os.environ.get("VIBEVOICE_VERIFY_WHISPER_MODEL", "")
    # Skip verification for very short inputs -- ASR similarity is unstable on a
    # few words, and a spurious retry costs a full synthesis.
    verify_min_chars: int = int(os.environ.get("VIBEVOICE_VERIFY_MIN_CHARS", "40"))

    # Max concurrent inferences per loaded model
    max_concurrency: int = int(os.environ.get("VIBEVOICE_MAX_CONCURRENCY", "2"))
    # Max seconds a request will wait for a free concurrency slot before giving
    # up server-side. Without this, a client that times out and disconnects
    # leaves its thread blocked on the semaphore (or running generate()) with
    # no way to reclaim the slot -- every abandoned request then permanently
    # competes for max_concurrency, and the thread pool backing sync routes
    # grows without bound since none of those threads ever return. This bounds
    # the wait (not in-progress generation -- see run_synthesis_pool below for
    # the thread-pool cap that bounds growth from requests already past this
    # point).
    queue_timeout_s: float = float(os.environ.get("VIBEVOICE_QUEUE_TIMEOUT", "240"))
    # Caps the worker thread pool used to run synthesis (sync FastAPI routes
    # run in Starlette's default threadpool, which is otherwise unbounded).
    # Small buffer above max_concurrency for requests waiting on the semaphore
    # above, so they can still queue and time out promptly rather than being
    # rejected outright by the pool itself.
    synthesis_pool_size: int = int(os.environ.get("VIBEVOICE_SYNTHESIS_POOL_SIZE", "0")) or (
        int(os.environ.get("VIBEVOICE_MAX_CONCURRENCY", "2")) + 4
    )
    # Logging directory & prompt logging
    logs_dir: str = os.environ.get("VIBEVOICE_LOG_DIR", os.path.join(os.getcwd(), "logs"))
    log_prompts: bool = os.environ.get("VIBEVOICE_LOG_PROMPTS", "1") not in {"0", "false", "False"}
    prompt_maxlen: int = int(os.environ.get("VIBEVOICE_PROMPT_MAXLEN", "4096"))
    instructions_maxlen: int = int(os.environ.get("VIBEVOICE_INSTRUCTIONS_MAXLEN", "2000"))
    # Default to system_only to prevent style text being spoken
    instructions_strategy: str = os.environ.get("VIBEVOICE_INSTRUCTIONS_STRATEGY", "system_only")
    instructions_repeat: int = int(os.environ.get("VIBEVOICE_INSTRUCTIONS_REPEAT", "1"))
    # SSE chunk aggregation bytes
    sse_chunk_bytes: int = int(os.environ.get("VIBEVOICE_SSE_CHUNK_BYTES", "16384"))
    # ffmpeg encoding knobs
    ffmpeg_bitrate: str = os.environ.get("VIBEVOICE_FFMPEG_BITRATE", "")  # e.g., "128k"
    flac_level: str = os.environ.get("VIBEVOICE_FLAC_LEVEL", "")  # 0-12
    # Opus container: webm (MSE-friendly) or ogg
    opus_container: str = os.environ.get("VIBEVOICE_OPUS_CONTAINER", "webm")
    # Opus VBR mode: vbr | constrained | cbr
    opus_vbr_mode: str = os.environ.get("VIBEVOICE_OPUS_VBR", "vbr")
    opus_application: str = os.environ.get("VIBEVOICE_OPUS_APPLICATION", "audio")  # audio|voip|lowdelay
    opus_frame_duration: str = os.environ.get("VIBEVOICE_OPUS_FRAME_DURATION", "")  # e.g., 20
    # AAC profile: aac_low | aac_he | aac_he_v2 | mpeg2_aac_low
    aac_profile: str = os.environ.get("VIBEVOICE_AAC_PROFILE", "")
    aac_mode: str = os.environ.get("VIBEVOICE_AAC_MODE", "cbr")  # cbr|vbr
    aac_q: str = os.environ.get("VIBEVOICE_AAC_Q", "")  # quality value for VBR (ffmpeg -q:a)
    # Auth
    # API key is DISABLED by default now. Set VIBEVOICE_REQUIRE_API_KEY=1 to enable.
    require_api_key: bool = os.environ.get("VIBEVOICE_REQUIRE_API_KEY", "0") not in {"0", "false", "False"}
    admin_token: str = os.environ.get("VIBEVOICE_ADMIN_TOKEN", "")
    keystore_path: str = os.environ.get("VIBEVOICE_KEYSTORE", os.path.join(os.getcwd(), "logs", "keys.json"))


CONFIG = ServerConfig()
