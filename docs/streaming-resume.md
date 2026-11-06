# Chat streams across a pyre restart

pyre's OpenAI `/v1/chat/completions` stream used to live only in the process. A crash, a VRAM
watchdog exit, or `srv.ps1` restarting the binary dropped the TCP connection and forgot every
token that had not already reached the client. Two things followed:

- The generation was gone. Nothing on disk could continue it.
- The client often did not notice. The socket close is easy to miss when a stream ends without
  `finish_reason`, and a wedged process that stops writing (including SSE comments) looks like a
  slow token. pi then sits on that socket until its own timeout, which for many OpenAI-compatible
  clients is "never" on a streaming read.

Successful streams still end with `finish_reason` `stop`, `length`, `tool_calls`,
`content_filter`, or `function_call`, then `data: [DONE]`. Every other ending is now explicit.

## What the server does

- **Request id on the headers and on every SSE chunk.** `x-request-id` and `x-resume-token`
  (the same `chatcmpl-...` value) are sent before the first token, so a death during prefill
  still names the journal. Each chunk's `id` is that value. `pyre.resume_token` repeats it.
- **Journal.** `PYRE_RESUME_DIR`, or `{temp}/pyre/generations`, gets one `{id}.jsonl` per chat.
  Each token is appended and flushed. The file holds the prompt token ids, sampling params, and
  the generated ids, so a resume continues the same sequence (the KV cache is gone; pyre
  re-prefills the prefix). `PYRE_RESUME=0` turns the journal off. `PYRE_RESUME_TTL_HOURS`
  (default 48, `0` keeps files) drops old journals at startup.
- **Startup recovery.** A journal with no closing line was killed mid-generation. The new
  process appends `status: aborted` ("server restarted before the generation finished").
- **Terminal frame, then `[DONE]`.** Failures send an OpenAI error object and a chunk with
  `finish_reason: "error"` and `pyre.status` of `aborted` or `error`. `[DONE]` is still written
  so a client blocked on the sentinel unblocks. `aborted` is resumable (disconnect, shutdown,
  crash). `error` is not (the model panicked or the prompt does not fit); replaying it will not
  start another generation.
- **SSE comments.** Every `PYRE_SSE_PING_SECS` (default 5) with no token, the server writes
  `: pyre-ping`. Set the client idle timeout to about three times that (15–20s).
- **Graceful drain.** SIGINT, SIGTERM, the VRAM watchdog's graceful stop (exit 87), and the
  hard floor exits (86 / 88) ask in-flight streams to emit `server shutting down` and wait
  `PYRE_DRAIN_MS` (default 2000, 400ms on the hard VRAM exits) before the process ends.
  `SIGKILL` cannot emit that frame; the journal and the client's idle timeout cover it.

A clean finish is not continued. Passing `continue: true` on a finished generation replays it
and stops. That keeps a retry from generating a second answer.

The prompt text is stored in the journal next to the server. Treat that directory like the
chat log. It is not copied anywhere else, and API keys are not written into it.

## Resume API

`GET /v1/generations/{id}`

```json
{
  "id": "chatcmpl-0123456789abcdef",
  "resume_token": "chatcmpl-0123456789abcdef",
  "status": "aborted",
  "finish_reason": "error",
  "message": "server restarted before the generation finished",
  "text": "partial raw model text",
  "resumable": true,
  "prompt_tokens": 1200,
  "completion_tokens": 40
}
```

`status` is `running`, `finished`, `aborted`, or `error`. `running` means this process still
holds the original stream (HTTP 409 on resume). After a restart it is `aborted`.

`POST /v1/chat/completions` with the original sampling ignored and:

```json
{ "resume": "chatcmpl-0123456789abcdef", "stream": true }
```

| Field | Default | Meaning |
| --- | --- | --- |
| `resume` | | The `x-resume-token` / chunk `id`. |
| `stream` | `false` | Same as a normal chat call. pi should send `true`. |
| `replay` | `true` | Emit the text already generated before any new tokens. Chunks that are replay carry `pyre.replay: true`. |
| `replay_from` | `0` | Byte offset into the **raw** model text the caller already has. The replay is the suffix. |
| `continue` | `true` only when `status` is `aborted` | Run the model for the remaining `max_tokens`. Ignored for `finished` and `error`. |

`POST /v1/generations/{id}/resume` is the same call with `stream: true`.

The continued prompt is the original prompt ids plus the generated ids. Sampling parameters are
the ones from the original request. A fixed seed does **not** reproduce the unsampled tail: the
RNG starts again at the seam. Greedy (`temperature: 0` or `top_k: 1`) does continue the same
sequence. The reply uses the same id, so the client can replace its in-progress assistant
message with the replay and then append.

While the original stream is still attached the resume returns **409** `generation_busy` and
the snapshot. Drop that stream (or let the server finish) and call again. A proxy should back
off briefly and retry a 409; closing the first connection aborts it and makes the journal
resumable.

## Client rules

These are the rules pi (or anything else speaking this HTTP API) has to follow. pyre does not
ship pi; `tools/pyre_resume_client.py` is a proxy that applies them in front of an unmodified
client.

1. **Connect timeout 10s.** A restarting server should fail fast, not hang the dial.
2. **Idle / read timeout 15–20s.** This must be a timeout on *silence*, not on the whole
   request. It has to be longer than `PYRE_SSE_PING_SECS` (default 5) so a quiet prefill, which
   still receives `: pyre-ping`, is not aborted. A dead process stops pinging and the read
   fails.
3. **Total timeout.** A ceiling for one user turn (30 minutes is enough for a long coding
   turn). It does not replace the idle timeout.
4. **A stream succeeded only if** some chunk has `finish_reason` in `stop`, `length`,
   `tool_calls`, `content_filter`, `function_call`, and you then see `data: [DONE]`.
   `finish_reason: "error"`, an `error` object, or EOF / timeout with no success reason is a
   failure even if `[DONE]` arrived.
5. **On failure, resume** with the `x-resume-token` (or the chunk `id`). Retry the resume a few
   times if the new process is not up yet (connection refused) or returns 409. Do not start a
   brand-new chat: that throws away the prefix and can duplicate a tool call.
6. **Dedupe replay.** With `replay: true` the server sends the prefix again, marked
   `pyre.replay`. Replace the assistant message, or forward only the suffix you have not shown.
   New tokens have `pyre.replay: false`.

## pi on the AN515, pyre on HYTE

Run the proxy on the machine where pi runs, aimed at the Windows box:

```bash
python tools/pyre_resume_client.py proxy \
  --upstream http://192.168.0.30:8099 \
  --listen 127.0.0.1:8100
```

If pyre was started with `--api-key`, the proxy forwards `Authorization` and `x-api-key`. Point
pi's OpenAI-compatible provider at the proxy, not at HYTE:

- base URL: `http://127.0.0.1:8100/v1`
- model id: the id `GET /v1/models` returns (the proxy forwards that)
- streaming: on (pi's openai-completions provider already streams)

No pi source change. The proxy's defaults are connect 10s, idle 20s, total 30 min, five resume
attempts. Override with `--connect-timeout`, `--idle-timeout`, `--total-timeout`,
`--resume-attempts`.

A one-shot check from the AN515, without pi:

```bash
python tools/pyre_resume_client.py complete \
  --upstream http://192.168.0.30:8099 \
  --model YOUR_MODEL_ID \
  --message "Say hello in one word."
```

Exit code 0 and a printed assistant message means the terminal event arrived. A non-zero exit
means the stream died and resume could not finish it; the stderr line says which.

## Tests without a GPU

`tools/pyre-resume` has no Burn dependency. From a host toolchain (the repo `.cargo/config.toml`
pins the Windows MSVC target used on HYTE; on another OS pass `--target` for that host):

```bash
cargo test --manifest-path tools/pyre-resume/Cargo.toml
```

That covers the journal (including a torn last line and crash recovery), the terminal frames,
a dropped worker, an engine error that must not be continued, a finished generation that must
not be generated twice, a graceful drain, and a kill-and-restart of a local server that
resumes the flushed prefix. Nothing in that crate loads a model.
