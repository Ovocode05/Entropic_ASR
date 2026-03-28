# Entropic ASR

A local, offline, agentic speech recognition pipeline for Hinglish — built for contexts where sending audio to a cloud API is not an option.

Whisper transcribes. DistilBERT normalizes numbers and classifies intent. A local LLM (Qwen 0.5B) runs a multi-turn conversation until it has everything it needs. The output is a structured JSON record. All of it runs on your own hardware.

---

## Why this exists

FIR statements in India are manually transcribed by constables. ASHA workers fill health forms from memory hours after a visit. Elderly property owners communicate inheritance wishes verbally — and courts spend years sorting out what was actually meant.

The common thread: structured information that needs to exist as a record, spoken by people who can't or won't type it, in a context where uploading audio to GPT-4 is either legally inadmissible, ethically wrong, or practically impossible.

Entropic ASR solves that. One sentence in Hinglish, structured JSON out.

---

## Pipeline

```
Voice Input (.wav)
      │
      ▼
┌─────────────────────┐
│  Whisper + LoRA     │  Stage 1 — ASR
│  (fine-tuned,       │  Romanised Hinglish output
│   whisper-small)    │  VAD trim applied pre-inference
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  DistilBERT ITN     │  Stage 2 — Numeric normalisation
│  Token classifier   │  "paanch hazaar" → 5000
│  (3-label: O/NUM/SEP│  Positional guard: "kar do" ≠ "kar 2"
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  DistilBERT Intent  │  Stage 3 — Intent + confidence
│  5-class classifier │  3-tier gate: ACCEPT / SOFT / HARD
│  + confidence gate  │  Temperature-scaled softmax
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Agent Decision     │  Stage 4 — Goal-directed conversation
│  Layer (Qwen 0.5B)  │  Tracks missing slots across turns
│  + SessionStore     │  Redis-backed, 4hr TTL
└────────┬────────────┘
         │
         ▼
  Structured JSON Record
```

**Per-stage latency (typical on DGX node, CUDA):**

| Stage | Time |
|-------|------|
| VAD + audio prep | ~7 ms |
| Whisper ASR | ~110 ms |
| DistilBERT ITN | ~2 ms |
| Intent classifier | ~2 ms |
| **Total** | **~120 ms** |

---

## Use cases

**FIR / Police statement**
Victim speaks in Hinglish. System extracts incident, amount, perpetrators, time, location, victim name across multiple turns. Verbatim transcript preserved alongside structured record. Audio never leaves the station's server.

**Oral asset declaration**
Elderly rural property owners speak their wishes. System produces a documented record of asset type, size, location, beneficiary, and declarant identity. Designed for zero literacy requirement.

**ASHA worker health records**
Community health workers dictate observations immediately after a home visit. System extracts child age, weight, symptoms, and household ID. Structured record synced to health system infrastructure.

---

## Privacy, safety & offline design

**Audio never leaves the machine.**
Every model in the stack — Whisper, DistilBERT ITN, DistilBERT Intent, Qwen 0.5B — runs on your own hardware. No API calls, no telemetry, no third-party servers. A victim's voice recording in an active FIR case has an evidentiary chain of custody. The moment that audio hits a commercial API endpoint, the chain breaks. This system is designed so that never happens.

**Cloud APIs are architecturally disqualified — not just impractical.**
Sending sensitive voice data to GPT-4 or Whisper API isn't a performance tradeoff. In legal, medical, and inheritance contexts it's a compliance failure. Entropic ASR runs on infrastructure the institution owns and controls. That's a non-negotiable requirement, not a nice-to-have.

**Verbatim preservation.**
Every turn is logged with its raw transcript, confidence score, and tier marker. The final record always includes the verbatim alongside the structured summary. If a record is ever challenged, you have the original words.

---

## Salient features

| Feature | Detail |
|---------|--------|
| **Hinglish-native** | Fine-tuned on Romanised Hinglish, not generic English ASR |
| **ITN with positional guard** | Converts "paanch hazaar" → 5000, preserves "kar do" as-is |
| **3-tier confidence gate** | ACCEPT / SOFT_REPROMPT / HARD_REPROMPT — no silent failures |
| **Keyword safety net** | If pipeline intent is uncertain, transcript keywords route to correct schema |
| **Per-stage latency** | VAD / ASR / ITN / Intent broken out per response — debuggable by design |
| **Redis session memory** | Survives server restarts, 4hr TTL, graceful in-memory fallback |
| **CUDA + CPU fallback** | Runs on GPU, auto-recovers to CPU on CUBLAS errors without crashing |
| **~120ms total latency** | On DGX node with CUDA — fast enough for real conversation |
| **Fully local LLM** | Qwen 0.5B for slot extraction and Hinglish question generation — no OpenAI |

---

## Confidence tiers

The pipeline doesn't silently accept bad transcriptions.

| Tier | Confidence | Behaviour |
|------|-----------|-----------|
| `ACCEPT` | ≥ 0.65 | Extract slots, advance conversation |
| `SOFT_REPROMPT` | 0.40–0.65 | Tentatively extract, ask for confirmation |
| `HARD_REPROMPT` | < 0.40 | Reject turn, ask to repeat. Escalates after 2 consecutive failures |

---

## Sample interaction

```bash
curl -X POST http://localhost:8000/chat \
  -F "session_id=fir_001" \
  -F "audio=@victim_statement.wav"
```

**Turn 1 response:**
```json
{
  "session_id": "fir_001",
  "pipeline_state": {
    "status": "ACCEPT",
    "transcript": "kal raat mere dukan mein ghuse, teen log the, paanch hazaar le gaye",
    "normalized_text": "kal raat mere dukan mein ghuse, teen log the, 5 hazaar le gaye",
    "intent": "FIR_THEFT",
    "confidence": 0.81,
    "latency": { "vad_ms": 7.1, "asr_ms": 108.3, "itn_ms": 2.1, "intent_ms": 2.0, "total_ms": 119.5 }
  },
  "agent_state": {
    "status": "incomplete",
    "missing_slots": ["location", "victim_name", "time"],
    "collected_slots": { "incident": "theft", "amount": "5000", "perpetrators": "3" },
    "agent_prompt": "Yeh kahan hua? Dukan ka address batao."
  }
}
```

**Turn 3 — all slots filled:**
```json
{
  "agent_state": {
    "status": "complete",
    "final_record": {
      "intent": "FIR_THEFT",
      "incident": "theft",
      "amount": "5000",
      "perpetrators": "3",
      "location": "Sharma General Store, Model Town, Amritsar",
      "victim_name": "Ramesh Sharma",
      "time": "last night",
      "total_turns": 3,
      "verbatim": [
        "[T1 | ACCEPT | conf=0.81]: 'kal raat mere dukan mein ghuse...'",
        "[T2 | ACCEPT | conf=0.78]: 'Model Town, Amritsar, Sharma General Store'",
        "[T3 | ACCEPT | conf=0.76]: 'Mera naam Ramesh Sharma hai'"
      ]
    },
    "eval_summary": {
      "total_turns": 3,
      "accepts": 3,
      "soft_reprompts": 0,
      "hard_reprompts": 0,
      "avg_confidence": 0.78
    }
  }
}
```

---

## Setup

```bash
git clone <repo>
cd Entropic_ASR
pip install -r requirements.txt
pip install streamlit streamlit-mic-recorder requests

# Start backend
python inference/api.py

# Start UI (separate terminal)
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

**Redis (optional, for session persistence across restarts):**
```bash
redis-server --daemonize yes
# Agent falls back to in-memory if Redis is unavailable
```

**Models required** (place in `models/adapters/`):
- `whisper_lora/` — fine-tuned Whisper small + LoRA adapter
- `distilbert_itn/` — ITN token classifier
- `distilbert_intent/` — intent classifier + `intent_config.json`

---

## Project structure

```
Entropic_ASR/
├── inference/
│   ├── api.py          # FastAPI — /chat and /transcribe endpoints
│   ├── pipeline.py     # Whisper → ITN → Intent, latency breakdown
│   └── agent.py        # Goal schema, session store, LLM extraction
├── models/adapters/
│   ├── whisper_lora/
│   ├── distilbert_itn/
│   └── distilbert_intent/
├── scripts/train/
│   ├── train_itn.py
│   └── train_intent.py
├── streamlit_app.py    # Demo UI
└── data/processed/
    └── financial_benchmark/
```

---

## What's next

- Retrain intent classifier on extended corpus (FIR, health, asset domain data)
- PDF export of final structured records
- Eval benchmark against human-transcribed ground truth

---
