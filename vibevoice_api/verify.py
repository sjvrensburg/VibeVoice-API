"""Post-synthesis verification: did the model actually say the text?

VibeVoice occasionally emits a garbled clip -- fluent, confident audio that is
not the requested text. It is rare and stochastic (measured 1/59 clips on a real
episode) and it is NOT triggered by particular input: the exact sentence that
failed in production re-synthesized cleanly 20 times out of 20. So it cannot be
prevented by choice of text, model size, or sampling -- the backbone already
runs greedy (do_sample=False), so there is no temperature to lower.

What it CAN be is detected. A clean clip transcribes back to its own input
almost exactly (similarity ~1.000); a garbled one scores far below. The gap is
wide, so the threshold is not delicate.

Deliberately shells out to whisper.cpp rather than importing an ASR stack: this
keeps the dependency optional and out of the server's import path, so a
deployment without it is unaffected rather than broken.
"""
from __future__ import annotations

import difflib
import logging
import re
import subprocess
import tempfile
import os
from typing import Optional, Tuple

log = logging.getLogger("vibevoice_api.verify")

_WORD = re.compile(r"[^a-z' ]+")


def _norm(s: str) -> str:
    return " ".join(_WORD.sub(" ", s.lower()).split())


def available(bin_path: str, model_path: str) -> bool:
    return bool(bin_path) and bool(model_path) \
        and os.path.isfile(bin_path) and os.access(bin_path, os.X_OK) \
        and os.path.isfile(model_path)


def transcribe(wav_bytes: bytes, bin_path: str, model_path: str,
               timeout: int = 300) -> Optional[str]:
    """Transcribe WAV bytes with whisper.cpp. None if it could not be run."""
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            fh.write(wav_bytes)
            tmp = fh.name
        proc = subprocess.run(
            [bin_path, "-m", model_path, "-f", tmp, "-nt"],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            log.warning("verify: whisper exited %s: %s", proc.returncode,
                        (proc.stderr or "")[-200:])
            return None
        return proc.stdout
    except Exception as exc:  # noqa: BLE001 - never let verification break synthesis
        log.warning("verify: transcription failed: %s", exc)
        return None
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def score(text: str, wav_bytes: bytes, bin_path: str, model_path: str) -> Optional[float]:
    """Similarity in [0,1] between the requested text and what was actually said.

    None means "could not check" -- callers must treat that as a pass, never as
    a failure, or a missing whisper binary would silently triple every request.
    """
    heard = transcribe(wav_bytes, bin_path, model_path)
    if heard is None:
        return None
    ref, hyp = _norm(text), _norm(heard)
    if not ref or not hyp:
        return None
    return difflib.SequenceMatcher(None, hyp, ref, autojunk=False).ratio()
