# Miso implementation checkpoint

This checkpoint distinguishes the implemented/tested block adapter from the
still-unqualified real model. See README for the current user-facing contract.

## Established upstream interface

- Pinned source: `MisoLabsAI/MisoTTS@cd7dde13dbfec33fd8321e6c34a43f86724a8bad`.
- `generator.load_miso_8b(device, model_path_or_repo_id, dtype)` constructs the
  model and `Generator`; `generate(text, speaker=0, context=[],
  max_audio_length_ms, temperature, topk)` returns a mono tensor at the Mimi
  sample rate (24 kHz). Initial integration does not require voice cloning.
- English only. Do not equate the historical backlog name “MISO ONE” with a
  separate verified model or copy hosted H100 latency into local claims.
- `Generator` currently obtains the tokenizer, Mimi and watermarker through
  implicit Hub calls. The owned worker must explicitly bind their loaders to the
  verified local files and set offline flags; a Run must never download anything.
  The upstream inference modules themselves are copied unchanged into each job.
- Preserve SilentCipher. `generator.MISO_TTS_WATERMARK` must receive the private
  deployment key (five integer bytes) through a private pipe, not argv, a blueprint,
  output metadata or logs. Resolve the secret through the public wallet service.
  `silentcipher.get_model` accepts explicit `ckpt_path` and `config_path`.
- `generate()` currently returns audio even if its frame budget is exhausted.
  Observe its `_model.generate_frame()` outputs to require a genuine all-zero EOS
  frame before publishing a final file. Do not advertise truncated speech as complete.
- The original loader allocates a float32 model before converting dtype. Loading
  peaks may exceed steady-state requirements: document and enforce resource bounds,
  not just parameter-count estimates. Native loading stays outside the framework.

## Implemented contract

1. Own `model.json`, `block.py` and a killable local worker. Use public
   `bloxsmith_app.block_api` only; no imports of other block implementations.
2. Inputs: fresh `text` and independent optional JSON `command` with
   `{"action":"interrupt"}`. Outputs: complete `audio_file`, correlated JSON
   `metadata`, recoverable `status`. Map by stable names, not list positions.
3. Active Runtime: public `listen_on_run` plus `runtime_listener` mailbox; one
   owned inference job and a bounded pending queue. Drain interrupt commands while
   inference runs, cancel the process group, discard pending work and fence late
   results. A Stop or parent death must not leave native model processes alive.
4. One Shot: actual bounded local synthesis on a text activation. Preparation is
   passive. Do not silently replay remembered text when the command input changes.
5. Output: private incomplete WAV while generating; publish a complete nonempty
   file atomically with a unique filename only after EOS, watermarking and format
   checks. Offer complete WAV/Ogg-Opus files, not fake audio-stream ports. Use node
   storage by default with an explicit optional output directory; never overwrite
   existing user files. Finalized-file retention and cancellation boundaries must
   be explicit.
6. EN/FR modal, inspector and node card. Shared draft configuration, Apply/Cancel,
   keyboard tabs, conditional fields and visible prerequisites. Declare every
   surface asset in the model and use release-scoped CSS and ES-module exports.
7. Test genuine protocol integration using bounded fixtures, all runtime/package
   origins, reversed ports, command interruption, Stop/parent death, file integrity,
   resource errors, secret redaction and responsive browser UI. Separately run the
   real pinned model on a host with prepared authorized assets and adequate resources.
   Never label a synthetic model fixture as real inference or model compatibility.
8. Only after applicable passing tests record exact-source compatibility. No commit,
   push, version increment, tag or release without explicit authorization.

## Current evidence and remaining gate

Three suites passed against BloxSmith 1.0.9 at framework commit
`f1a87f6369d2acfb8be41d23ea849384cd461c86` in the private report
`tests/results/run-mwdda2yx/report.json` (83.751 seconds). Exact-source compatibility
was recorded in `compatibility.json` at block version 0.1.0. This is **fixture-based
model protocol qualification**, not actual Miso inference.

The tests include eight asset/helper cases, three deterministic listener cases,
local loader arguments, EOS gating, real FFmpeg WAV/Opus round trips, native-process
cancellation/timeout/parent death, atomic publication, actual disposable wallets,
all six runtime/package-origin combinations and responsive EN/FR browser surfaces.
Modal screenshots at 320/390/800/1440 widths, advanced settings and inspector were
reviewed. Normal jobs clean up; forced managed-host termination can retain private
unreferenced job directories, explicitly documented rather than claimed absent.

The six small public engine files and
SilentCipher YAML match their actual upstream Git blob hashes. Public weights are
pinned by their upstream LFS SHA-256 metadata, not yet downloaded or executed here.
The read-only helper reported 33,158,100,652 required bytes and approximately
5.3 GB free. The Llama tokenizer's official repository requires manual access
approval; no credentials or authorized files were provided to this work.

Real-model testing needs the prepared authorized assets, dedicated environment,
adequate host RAM/VRAM and a private deployment watermark key. The optional
`tests/qualify_model.py` is read-only by default and offers explicitly enabled real
smoke inference. Actual generation/listening, watermark review and runtime Stop
qualification still remain. These are resource/access gates, not framework gaps.

The independent public HackInvent repository is empty. The manifest, runtime,
process guardian, native worker adapter and responsive bilingual UI now exist.
Transparent artwork and English documentation are present. No source commit,
push, version increment, tag or release was made. The full backlog item is not
marked complete while actual Miso execution remains unqualified.
