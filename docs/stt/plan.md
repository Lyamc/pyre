# STT follow-up plan

Checklist for the quality gaps left after PR #13 (`qwen-int` at `02189de`). Docs only. Do not change STT, TTS, Qwen-Next, or context code in the planning PR.

Lyam’s HYTE tree is `C:\Temp\qnx\stt13` (`results.csv`, `run6.ps1`, `score6.py`). That path is not on the machine that wrote this plan, so the WER figures below are the ones he reported. The fix PRs have to re-score with those scripts, or say that HYTE was not available. Do not invent a WER.

The gate that has to stay until the criteria at the bottom are met:

- `src/speech_caps.rs`: `gate_experimental_model` and `gate_long_audio`. Parakeet TDT v3, Nemotron Speech Streaming, and Fluid Speech Pro are `experimental: true`. `SHORT_CLIP_SECS` is 60.
- `crates/pyre-whisper-engine/src/transcribe/mod.rs`: `experimental_gate` for the CLI. v3, Nemotron, and Fluid Speech Pro are refused unless `PYRE_STT_EXPERIMENTAL` is `1`, `true`, `yes`, or `on`. Other Parakeet ids refuse audio longer than 60 s on that same switch.
- Opt-in is also `experimental=true` on the HTTP request (`speech_caps::experimental_opt_in`).

## Reported symptoms

- [x] Record the eight items. Numbers are from Lyam, not re-measured here.
- [ ] Copy `results.csv` rows for the failing clips into the fix PR (utterance id, model, hypothesis, reference, WER). The planning environment could not read `C:\Temp\qnx\stt13`.

| # | Symptom | Reported figure |
| --- | --- | --- |
| 1 | Parakeet TDT v3 on French | about 70% WER, empty hypotheses |
| 2 | Nemotron Speech Streaming 0.6B | about 16% WER, substitutions at the start of utterances |
| 3 | One Parakeet v2 utterance | hypothesis is all caps |
| 4 | Whisper | a repeated phrase (`come come…`) disappears at a chunk boundary |
| 5 | `pyre-whisper-engine` unit tests | 3 tests were already failing before PR #13 |
| 6 | v3 and Nemotron load | f32 only, about 2.4 GB each |
| 7 | Audio longer than 60 s on the new models | gated |
| 8 | `PYRE_STT_EXPERIMENTAL` | still required for v3 and Nemotron |

## 1. Parakeet TDT v3 French

Code to read before changing anything:

- `crates/pyre-whisper-engine/src/model/parakeet.rs`, `decode_tdt_greedy`. The decoder starts on the blank id. A comment there says a `<|startofcontext|>` or language prompt made v3 invent a prefix, and PR #13 commit `a64fb31` reverted “the v3 prompt that added text”.
- `crates/pyre-whisper-engine/src/transcribe/decode/parakeet.rs`. SentencePiece detokenization (`keep_parakeet_piece`, byte pieces `<0xC3>`, apostrophes). Control tokens are stripped at detokenize time. The argmax still includes them, on purpose: masking them changes duration and LSTM state.
- `src/speech_caps.rs`: v3’s language list includes `fr`. The profile says v3 does not translate.

Checks:

- [x] 2026-11-14: the shared TDT decoder (`parakeet_decoder.rs`) already starts blank as a zero vector, not `embedding[blank]`. A comment there says feeding the embedding made v3 emit blank plus a long duration and an empty utterance. Fluid Speech Pro uses that same decoder and the v3 tokenizer. This environment did not re-score French and did not run NeMo. The empty-output checkboxes below stay open.
- [ ] On one French clip from `stt13`, log token ids, blank steps, duration indices, and the string after detokenize. Separate “decoder emitted nothing” from “decoder emitted control or byte pieces that became an empty string”.
- [ ] Compare those token ids to a NeMo `GreedyBatchedTDTInfer` run on the same waveform and the same checkpoint. Record both transcripts. That is the parity check.
- [ ] Confirm the SentencePiece model is the v3 vocab, not the v2 vocab (vocab size, blank id, pad id, byte pieces). `ParakeetDecodeConfig` defaults pad to 2 and, when `decoder_start_token_id` is missing, starts at blank. Check the v3 decode JSON on disk against that.
- [ ] Language handling: v3 is supposed to take the language from the audio. Verify the HTTP `language=fr` path does not feed a prompt token, and does not silently route the clip to Whisper (`speech_caps` routing). If the empty outputs are only on the routed path, fix routing rather than the TDT loop.
- [ ] Do not put the reverted language prompt back unless the NeMo reference itself conditions on that token and a before/after on the French set shows fewer empties and no invented prefix.
- [ ] Re-score French with `score6.py`. Target is in the removal criteria below, not “looks better”.

## 2. Nemotron Speech Streaming 0.6B

Code:

- `crates/pyre-whisper-engine/src/model/nemotron.rs`: `chunk_mask`, `sliding_window`, lookahead, `nemotron_pos_embed`. The speech profile sets `lookahead_frames: Some(13)` and `architecture_streaming: true`.
- `a64fb31` reverted “the Nemotron pad”. Read that diff before adding padding again. The revert was because the pad added text.

Checks:

- [ ] From `results.csv`, list substitutions that sit in the first words. Note whether the reference word is deleted, replaced, or prefixed.
- [ ] Compare left context, cache init, and the first chunk’s valid length against the NeMo streaming config for this checkpoint (chunk size, left context, lookahead). Log the values pyre actually uses.
- [ ] Run the same utterances through the reference. Parity is the reference transcript, not a lower WER against a different normalizer.
- [ ] If the bug is the reverted pad, try a pad that the reference uses and show it does not add the extra words `a64fb31` was undoing.
- [ ] Re-score with `score6.py`.

## 3. Parakeet v2 all-caps utterance

Code: `pieces_to_text` keeps the case of the SentencePiece pieces (`▁O` + `ther` → `Other`). `clean_text` does not uppercase.

Checks:

- [ ] Find the one utterance in `results.csv`. Record hypothesis, reference, and model id.
- [ ] Decode it again with token ids printed. If the pieces are already uppercase, the bug is the model or the wrong tokenizer, not a post-process. If the pieces are mixed case and the HTTP or CLI string is upper, find the uppercasing call.
- [ ] Add a regression test only if it can be a tiny fixture (pieces in, text out). Do not check in the audio.

## 4. Whisper repeated phrase at a chunk boundary

Code:

- `crates/pyre-whisper-engine/src/transcribe/mel.rs`: chunks overlap by 3 s (`chunk_overlap = sample_rate * 3`).
- `crates/pyre-whisper-engine/src/transcribe/utils.rs`: `append_without_word_overlap` drops a case-insensitive word suffix/prefix of up to 40 words. `drop_left_context_tokens` drops tokens whose frame is before `min_frame`. `timestamps_collapsed` is the escape hatch when that drop keeps less than half of a long window.

Why this matches “come come”: if the previous window ends in `come` and the next window starts with `come`, a one-word match treats the second `come` as overlap and deletes it, even when the speaker repeated the word across the boundary.

Checks:

- [ ] Locate the clip and the chunk index in the HYTE log. Confirm the two `come` tokens belong to different windows.
- [ ] Change the stitch so a single repeated word is not enough to drop a token. Prefer frame overlap (`drop_left_context_tokens`) when timestamps exist, and keep the word stitch only as the fallback `timestamps_collapsed` already describes.
- [ ] Regression test on strings: `… come` plus `come come …` keeps both words; a real 3-word overlap (`can do for you`) is still dropped once. The existing unit test in `utils.rs` covers the 3-word case and must stay green.
- [ ] Re-run that Whisper clip. Do not retune the stitch on v3 or Nemotron in the same change unless the same function is why they drop edge words, and say so.

## 5. Three `pyre-whisper-engine` unit tests

- [ ] On a checkout of the parent of PR #13, run `cargo test -p pyre-whisper-engine --offline` (or the command HYTE used) and save the three failing names. They were already red before that PR, so the fix is not “revert #13”.
- [ ] This plan does not name them. The planning environment did not run the suite, and guessing the names would be wrong.
- [ ] Fix those three, or adjust the assertion if the test is stale, with a note of which. Do not weaken a parity test to hide item 1 or 2.
- [ ] The fix PR lists the three names and the `cargo test` summary.

## 6. f16 / bf16 or quantized load

Code: `crates/pyre-whisper-engine/src/model/load/helpers.rs` loads `NpyData::<f32>` only. A 0.6B model in f32 is about 2.4 GB of weights, which matches the report. Parakeet and Nemotron both come in through npy under the model directory (`load_parakeet` and the Nemotron loader).

Checks:

- [x] 2026-11-14: `decode_npy_f32` in `helpers.rs` widens float16 and bfloat16 npy to f32 while loading. The conformer and LSTM graph stay f32, and the served `model.bpk` is still f32 (about 2.4 GB). This VM has no NVIDIA GPU, so half matmul and a VRAM figure were not measured. STT stays on CPU, which leaves the HYTE 12 GB card for the Qwen weights.
- [ ] Store or convert weights as f16 or bf16 and cast per op, or keep them packed if Burn on this fork can run the matmul. Follow the TripoSR pattern (f16 on device, f32 math) only if the conformer and LSTM match that. Quantized load is acceptable if it is the same packing pyre already uses for GGUF, and only with a measured WER.
- [ ] Conversion stays in Rust. No Python at runtime.
- [ ] GPU test: load v3 and Nemotron in the new dtype on a CUDA device, transcribe one clip from `stt13`, and compare the text to the f32 run of the same clip. Record peak memory if the tool prints it. If no GPU is available, say so and do not claim a memory number.
- [ ] f32 remains the fallback if the half file is missing.

## 7. Ungate audio longer than 60 s

The chunker already exists (`waveform_to_mel_tensor`, 3 s overlap, `append_without_word_overlap`, `drop_left_context_tokens`). `gate_long_audio` and `experimental_gate` refuse it for Parakeet-family ids unless the experimental switch is on. The comment in `speech_caps.rs` says joined audio still loses or repeats words at window edges.

Plan the windowing, then ungate:

- [ ] Write down the window length, hop, and overlap the new models should use. Whisper’s 3 s overlap is not automatically right for TDT or for Nemotron’s chunk mask. Take the values from the NeMo streaming config when the reference has them.
- [ ] Fix item 4’s stitch, and item 2’s start-of-utterance bug, before trusting a long file. A long file is a sequence of those edges.
- [ ] Score at least one clip over 60 s, with the gate forced on, against the reference. Record insertions and deletions at each boundary, not only the file WER.
- [ ] Ungate only that model’s long path (`gate_long_audio` / the `secs > 60` branch) when the boundary errors are gone on that clip set. Leave the model-level experimental gate (item 8) independent: a good window does not by itself make v3 French acceptable.

## 8. When to remove `PYRE_STT_EXPERIMENTAL`

Remove `experimental: true` for a model only after a fix PR pastes a new `score6.py` table from HYTE (or states that the run could not be repeated). Suggested bar, to be confirmed against the model card rather than invented here:

- [ ] Parakeet v3: on the `stt13` French set, empty hypotheses are rare (write the count, not a vibe), WER is in the range of the NeMo reference on those same files, and a spot check of another v3 language does not show the invented prefixes `a64fb31` removed. Look up the card’s published WER and quote it in the PR. “About 70%” is not the bar.
- [ ] Nemotron: on the `stt13` set, WER is in range of the NeMo reference, and the start-of-utterance substitutions in item 2 are gone or counted. Quote the card’s published number in the PR. “About 16%” is the current pyre figure, not the target.
- [ ] Items 3, 4, and 5 are fixed or explicitly waived in the PR.
- [ ] Item 6 is not required to clear the gate. Memory format and transcript quality are separate switches.
- [ ] Item 7 can clear the long-audio gate for one model while the other model stays experimental.
- [ ] After removal, `speech_caps` tests that currently expect `experimental: true` for that id are updated in the same PR, and Whisper plus Parakeet v2 tests stay non-experimental.

## PR split

- [x] This file, in the planning PR, with `docs/3d/plan.md`.
- [ ] PR — items 1 and 3 (v3 French and the v2 casing), plus any of the three unit tests that are in the Parakeet decoder.
- [ ] PR — item 2 (Nemotron edges).
- [ ] PR — item 4 (Whisper stitch) with the string regression.
- [ ] PR — item 5 for whichever of the three tests are not Parakeet.
- [ ] PR — item 6 (dtype), after a GPU or an explicit “no GPU” note. The fluid-speech-pro PR records the no-GPU note and the f32-widening loader. It does not close this item.
- [ ] PR — item 7, then a one-line change to the long-audio gate once the score exists.
- [ ] PR — item 8, only a flag change plus the pasted scores. No drive-by decode edits.

Keep these off the 3D branches. No force-push.

## Open questions

- [ ] Are the 70% and 16% figures micro-average WER from `score6.py`, and which language subset?
- [ ] What are the three failing test names on HYTE?
- [ ] Is the NeMo reference already scripted in `run6.ps1`, or does the fix PR have to add that comparison?
- [ ] Which published WER (model card vs. an internal set) is the bar for item 8?

## Fluid Speech Pro (added on qwen-int, not a scored fix)

The model id is `fluid-speech-pro` (`FluidVoice-Labs/fluid-speech-pro`, CC-BY-4.0). It is Parakeet TDT v3 plus the NeMo `encoder.fv_insert` block and a clean-mode LoRA. The Hugging Face safetensors omit that block, so conversion reads `fluid-speech-pro.nemo`.

- [x] Extra block and clean mode live on `#[module(skip)]` fields so an existing v2/v3 `model.bpk` still loads. Sidecars sit under `fluid/` next to the pack.
- [x] Clean mode is a request flag (`clean_mode`), not a second set of weights. Beam 1..=4 is only applied when the model id contains `fluid-speech`.
- [x] Word spans come from TDT frame indices at 80 ms, grouped on SentencePiece `▁`.
- [ ] WER against LibriSpeech or FLEURS, including French and German, side by side with v2 and v3. This environment had no GPU and did not run the 2.6 GB checkpoint, so there is no table. The model card’s Open ASR average (Fluid 6.06, v2 6.93, v3 7.04) is the card’s number, not a pyre measurement. The experimental gate stays on until a HYTE score exists.
- [ ] Logit parity against NeMo on a short clip. Unit tests cover the LoRA delta, the bias fold, and the frame-insert residual against the published formula. They are not a transcript fixture.
