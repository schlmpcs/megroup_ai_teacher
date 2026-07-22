# Kazakh Calm Teacher Voice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all Kazakh OmniVoice synthesis use a calm teacher delivery and deploy it to production.

**Architecture:** Keep the existing fixed-profile OmniVoice service, use its supported low-pitch young-male profile, and send Kazakh OmniVoice requests at speed `0.9`. Mirror the profile in Docker Compose, then rebuild the production API and `voice-omnivoice` containers.

**Tech Stack:** Python, FastAPI, pytest, Docker Compose, OmniVoice, ffmpeg

## Global Constraints

- Keep the existing young male Kazakh voice.
- Use the supported profile `male, young adult, low pitch`.
- Send Kazakh OmniVoice requests at speed `0.9`; keep other backends at `1.0`.
- Do not add per-request intonation controls or dependencies.
- Do not restart unrelated production services.

---

### Task 1: Change And Verify The Default Profile

**Files:**
- Modify: `tests/test_tts_text_normalization_backends.py`
- Modify: `tests/test_voice_units.py`
- Modify: `voice_omnivoice/app/main.py`
- Modify: `app/services/voice.py`
- Modify: `docker-compose.yml`

**Interfaces:**
- Consumes: `voice_omnivoice.app.main.Settings.instruct`
- Produces: the calm teacher instruction used by OmniVoice at startup

- [ ] **Step 1: Update the existing instruction assertion**

```python
assert call["instruct"] == "male, young adult, low pitch"
assert fake.calls[0][1]["json"]["speed"] == 0.9
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `pytest tests/test_tts_text_normalization_backends.py::test_omnivoice_model_receives_normalized_kazakh_text -q`

Expected: FAIL because the current profile is unsupported and the request speed is `1.0`.

- [ ] **Step 3: Change both defaults**

Set `Settings.instruct` and the Compose `OMNIVOICE_INSTRUCT` fallback to:

```text
male, young adult, low pitch
```

- [ ] **Step 4: Run verification**

Run:

```bash
pytest tests/test_tts_text_normalization_backends.py::test_omnivoice_model_receives_normalized_kazakh_text -q
pytest
docker compose config | grep 'OMNIVOICE_INSTRUCT: male, young adult, low pitch'
```

Expected: all tests pass and Compose renders the new profile.

- [ ] **Step 5: Commit and push**

```bash
git add tests/test_tts_text_normalization_backends.py tests/test_voice_units.py voice_omnivoice/app/main.py app/services/voice.py docker-compose.yml docs/superpowers/plans/2026-07-22-kazakh-calm-teacher-voice.md docs/superpowers/specs/2026-07-22-kazakh-calm-teacher-voice-design.md
git commit -m "feat: use calm teacher Kazakh voice"
git push origin main
```

### Task 2: Deploy And Regenerate The Kazakh Introduction

**Files:**
- Replace locally: `generated_audio/assistant_intro_kk.mp3`

**Interfaces:**
- Consumes: production `POST /tts/synthesize?format=wav`
- Produces: healthy production Kazakh TTS and the updated local MP3 sample

- [ ] **Step 1: Update and rebuild only OmniVoice**

```bash
ssh megroup-b560m-hdv-m-2 'cd /home/megroup/megroup_ai_teacher && git pull --ff-only && docker compose up -d --build api voice-omnivoice'
```

- [ ] **Step 2: Verify the production profile and health**

Run: `curl --fail http://megroup-b560m-hdv-m-2:8003/health`

Expected: status `ok`, model loaded, and the new calm teacher profile.

- [ ] **Step 3: Synthesize the introduction and convert it to MP3**

POST the approved Kazakh introduction with `speed=1.0` and `backend=omnivoice`, save the WAV, then run:

```bash
ffmpeg -y -i generated_audio/assistant_intro_kk.wav -codec:a libmp3lame -b:a 192k generated_audio/assistant_intro_kk.mp3
```

- [ ] **Step 4: Verify the generated audio**

Run: `ffprobe -v error -show_entries stream=codec_name,sample_rate,channels:format=duration,size generated_audio/assistant_intro_kk.mp3`

Expected: a non-empty mono MP3 with positive duration.
