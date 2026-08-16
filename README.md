# VibeVoice API

> This is an OpenAI-compatible API for VibeVoice.

## This fork

Fork of [vibevoice-community/VibeVoice-API](https://github.com/vibevoice-community/VibeVoice-API)
with a fix for a production issue found while running this server behind
[Open Notebook](https://github.com/lfnovo/open-notebook) on concurrent podcast-generation
traffic: **abandoned requests were never cancelled server-side, and each one leaked a thread.**

`synthesize()` and `synthesize_stream_pcm()` (`vibevoice_api/tts_engine.py`) acquired the
concurrency semaphore with a plain blocking `with state.semaphore:`. A client that gave up after
its own timeout left the server with no way to know — the request kept running (or waiting) to
completion regardless, permanently holding both its thread (FastAPI's default thread pool for sync
routes is otherwise unbounded) and, while running, the concurrency slot. Under real concurrent
load this compounds: two back-to-back batches of 5 concurrent requests pinned a GPU at 99% for over
30 minutes and grew the server's thread count from a ~95-thread baseline to 356, never dropping —
recoverable only by restarting the process.

Fixed in [`e32cd9b`](https://github.com/sjvrensburg/VibeVoice-API/commit/e32cd9b901da183b4e1d1db69cfba5c51c76a0cf):

- `state.semaphore.acquire(timeout=CONFIG.queue_timeout_s)` replaces the unbounded form in both
  the sync and streaming synthesis paths. A request that can't get a slot in time raises a new
  `QueueTimeoutError` and releases its thread immediately instead of waiting indefinitely. New env
  var: `VIBEVOICE_QUEUE_TIMEOUT` (default `240` seconds).
- `server.py` maps `QueueTimeoutError` to `503` + `Retry-After`, distinct from a `500` synthesis
  failure, so callers can tell transient load from a broken request.
- The streaming (SSE) path calls `AsyncAudioStreamer.end()` on a pre-generate failure so its
  consumer loop can't hang waiting for a stop signal that would otherwise never arrive.
- App startup caps `anyio`'s default thread limiter to `VIBEVOICE_SYNTHESIS_POOL_SIZE` (default
  `max_concurrency + 4`), bounding how many sync-route worker threads can exist concurrently in
  the first place — this is what actually stops orphaned requests from being *able* to accumulate
  a thread each, complementing the queue timeout rather than duplicating it.

Verified against a real reproduction: 5 concurrent long-clip requests at
`VIBEVOICE_MAX_CONCURRENCY=1`, 3 correctly fail with a clean `503` at ~246s instead of hanging, the
other 2 succeed normally, and the thread count returns to a stable steady state afterward instead
of climbing indefinitely.

## Community

**Join the unofficial Discord community: https://discord.gg/ZDEYTTRxWG** - share samples, ask questions, discuss fine-tuning, etc.

## Installation

```bash
git clone https://github.com/vibevoice-community/VibeVoice-API
cd VibeVoice/

uv pip install -e .
```

## Model Zoo

| Model | Context Length | Generation Length |  Weight |
|-------|----------------|----------|----------|
| VibeVoice-1.5B | 64K | ~90 min | [HF link](https://huggingface.co/vibevoice/VibeVoice-1.5B) |
| VibeVoice-Large| 32K | ~45 min | [HF link](https://huggingface.co/vibevoice/VibeVoice-7B) |

## Getting Started

Run a local server that is compatible with the OpenAI audio API (`client.audio.speech.create`). It wraps VibeVoice to synthesize speech from text.

### Start the server
```bash
python -m vibevoice_api.server --model_path vibevoice/VibeVoice-1.5B --port 8000
```

### API base path (default: `/v1`)
All routes are mounted on `/v1` by default. To override the prefix, set `VIBEVOICE_API_BASE_PATH` (leading slash required) **before** launching the server:
```bash
export VIBEVOICE_API_BASE_PATH=/api
python -m vibevoice_api.server --model_path vibevoice/VibeVoice-1.5B --port 8000
```
Clients must include the same prefix when constructing URLs. The static console is served at `<base_path>/web/console.html`.

## Endpoints

### POST `<base_path>/audio/speech`
Synthesize speech from text.

**Request fields (OpenAI-compatible):**
- `model` (string): model id or local path (e.g., `vibevoice/VibeVoice-1.5B`).
- `voice` (string): name mapped to a reference voice, a filesystem path (prefix with `path:` or absolute), or an alias from a voice map.
- `input` (string): the input text.
- `response_format` (string): `wav`, `pcm` (native), or `mp3` / `opus` / `aac` (require ffmpeg).
- `stream_format` (string, optional): set to `sse` for Server-Sent Events (streamed base64 PCM chunks).
- `extra_body` (object, optional):
  - `voice_path`: absolute/relative path to a reference audio file.
  - `voice_data`: base64-encoded WAV bytes (optionally as a data URL).

**Python example (OpenAI SDK ≥ 1.40):**
```python
from openai import OpenAI

base_path = "/v1"  # or your VIBEVOICE_API_BASE_PATH
client = OpenAI(base_url=f"http://127.0.0.1:8000{base_path}", api_key="<YOUR_API_KEY>")

speech = client.audio.speech.create(
    model="vibevoice/VibeVoice-1.5B",
    voice="Andrew",
    input="Hello from VibeVoice!",
    response_format="wav",
)

with open("out.wav", "wb") as f:
    f.write(speech.read())
```

**Pure HTTP example (cURL):**
```bash
curl -X POST "http://127.0.0.1:8000/v1/audio/speech"   -H "Content-Type: application/json"   -H "Authorization: Bearer <YOUR_API_KEY>"   -d '{
    "model": "vibevoice/VibeVoice-1.5B",
    "voice": "alloy",
    "input": "Hello!",
    "response_format": "mp3"
  }' --output out.mp3
```

**Streaming (SSE):**
Set `"stream_format": "sse"` in the request body to receive a stream of SSE events carrying base64-encoded PCM audio chunks. A JS example client is provided in `scripts/js/openai_sse_client.mjs`.

## Voice Mapping

You can define stable, human-friendly voice names via a YAML file that is auto-loaded on each request.
- Voice YAML mapping: You can use YAML to manage aliases or automatically scan multiple folders (see next section).

**Search order (first found):**
1. Path from `VIBEVOICE_VOICE_MAP` (relative to repo root or absolute)  
2. `./voice_map.yaml`  
3. `./config/voice_map.yaml`

**Example (`voice_map.yaml`):**
```yaml
alloy: en-Frank_man
ash: en-Carter_man

aliases:
  promo_female: demo/voices/en-Alice_woman.wav

directories:
  - demo/custom_voices
```

Then call with `voice: "alloy"`, or use `extra_body.voice_path` / `extra_body.voice_data` per request.

## Formats

- `wav`, `pcm`: native outputs (no extra dependencies).
- `mp3`, `opus`, `aac`: require a working **ffmpeg** binary. Either ensure `ffmpeg` is on PATH or set `VIBEVOICE_FFMPEG` to the binary path.

## Authentication & Admin (optional)

By default, API-key auth is **disabled**. To enable:
```bash
export VIBEVOICE_REQUIRE_API_KEY=1
```

With auth enabled, include `Authorization: Bearer <YOUR_API_KEY>` in client requests.

**Admin key management** (requires `VIBEVOICE_ADMIN_TOKEN`; routes respect your `<base_path>` and default to `/v1`):

_List stored key hashes_
```bash
curl -sS -H "Authorization: Bearer $VIBEVOICE_ADMIN_TOKEN"   http://127.0.0.1:8000/v1/admin/keys
```

_Create/import a key (omit body to auto-generate with the given prefix)_
```bash
curl -sS -X POST -H "Authorization: Bearer $VIBEVOICE_ADMIN_TOKEN"   -H "Content-Type: application/json"   -d '{"prefix": "sk-"}'   http://127.0.0.1:8000/v1/admin/keys
```

_Revoke a key by stored hash_
```bash
curl -sS -X DELETE -H "Authorization: Bearer $VIBEVOICE_ADMIN_TOKEN"   http://127.0.0.1:8000/v1/admin/keys/<key_hash>
```

**Logs** are written under `logs/` and can be configured via:
- `VIBEVOICE_LOG_DIR`
- `VIBEVOICE_LOG_PROMPTS=1`
- `VIBEVOICE_PROMPT_MAXLEN=4096`

## Notes
- Only TTS (`/audio/speech`) is implemented; there are **no STT endpoints**.
- Legacy root routes (e.g., `/audio/speech`, `/metrics`) remain for backwards compatibility, but new integrations should prefer the explicit `<base_path>`.


## License

The source code and models are licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

Note: Microsoft has removed the original repo and models. This fork is based off of the MIT-licensed code from Microsoft.
