"""
agent.py  —  SmartAgentDecisionLayer  (LLM-first redesign)
------------------------------------------------------------
The LLM is the orchestrator, not just a slot extractor.

Old design:  code decides tier / follow-up → LLM extracts slots
New design:  LLM receives full context → LLM decides everything
             (what was extracted, what is missing, what to ask next,
              whether we are done, how to handle unclear audio)

GoalSchemas are kept as REFERENCE HINTS in the system prompt so the
LLM knows the domain — they are NOT enforced in code anymore.
"""

import json
import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Redis session store ───────────────────────────────────────────────────────
try:
    import redis
    _redis = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    _redis.ping()
    REDIS_AVAILABLE = True
    print("  [SESSION] Redis connected.")
except Exception:
    REDIS_AVAILABLE = False
    print("  [SESSION] Redis unavailable — using in-memory fallback.")

SESSION_TTL = 60 * 60 * 4   # 4 hours


class SessionStore:
    def __init__(self):
        self._mem: dict = {}

    def _key(self, sid: str) -> str:
        return f"entropic:session:{sid}"

    def get(self, sid: str) -> dict | None:
        if REDIS_AVAILABLE:
            raw = _redis.get(self._key(sid))
            return json.loads(raw) if raw else None
        return self._mem.get(sid)

    def set(self, sid: str, data: dict):
        if REDIS_AVAILABLE:
            _redis.setex(self._key(sid), SESSION_TTL, json.dumps(data, ensure_ascii=False))
        else:
            self._mem[sid] = data

    def delete(self, sid: str):
        if REDIS_AVAILABLE:
            _redis.delete(self._key(sid))
        else:
            self._mem.pop(sid, None)


# ── System prompt ─────────────────────────────────────────────────────────────
# CRITICAL: Qwen 0.5B follows JSON format best when it sees CONCRETE EXAMPLES.
# Abstract instructions alone cause it to generate generic English.
# Few-shot examples are baked in → model knows exactly what output looks like.
SYSTEM_PROMPT = """You are Entropic, an intelligent Hinglish voice assistant for India.
You collect structured information through conversation, speaking in Hinglish (Hindi-English mix).

SUPPORTED INTENTS AND THEIR REQUIRED FIELDS:
- SEND_MONEY     → amount, recipient
- CHECK_BALANCE  → account_type
- BILL_PAYMENT   → bill_type, amount, biller_name
- RECEIVE_MONEY  → amount, sender
- EXPENSE_LOG    → amount, category
- FIR_THEFT      → incident, amount_stolen, num_perpetrators, time, location, victim_name
- FIR_ASSAULT    → incident, time, location, victim_name, accused_description
- ASSET_DECLARATION → asset_type, size_or_value, location, beneficiary, declarant_name
- HEALTH_RECORD  → child_age, weight, symptom, household_id
- UNKNOWN        → clarify use case before collecting any fields

YOUR OUTPUT MUST BE VALID JSON, EXACTLY IN THIS SCHEMA:
{"extracted":{"field":"value"},"all_collected":{"field":"value"},"still_needed":["field1"],"status":"incomplete","hinglish_response":"your message"}

RULES:
1. hinglish_response MUST be in Hinglish (mix Hindi+English words). NEVER pure English.
2. Extract ONLY what user explicitly said. Never invent values.
3. Ask ONE question per turn for the MOST IMPORTANT missing field.
4. If all fields collected → set status to "complete".
5. If intent is UNKNOWN → ask what the user wants to do, in Hinglish.
6. If audio_tier is HARD_REPROMPT → ask user to repeat clearly, in Hinglish.
7. No markdown, no explanation, ONLY JSON.

FEW-SHOT EXAMPLES (follow these exactly):

Example 1 — SEND_MONEY, first turn, amount known, recipient missing:
Input: transcript="1000 rupay Rahul ko bhej do", intent="SEND_MONEY", already_collected={}
Output: {"extracted":{"amount":"1000","recipient":"Rahul"},"all_collected":{"amount":"1000","recipient":"Rahul"},"still_needed":[],"status":"complete","hinglish_response":"Theek hai! 1000 rupay Rahul ko transfer kar rahe hain. Confirm karein?"}

Example 2 — SEND_MONEY, recipient missing:
Input: transcript="paanch hazaar bhejo", intent="SEND_MONEY", already_collected={"amount":"5000"}
Output: {"extracted":{},"all_collected":{"amount":"5000"},"still_needed":["recipient"],"status":"incomplete","hinglish_response":"Paisa kis ko bhejna hai? Naam batayein."}

Example 3 — UNKNOWN intent (greeting/unclear):
Input: transcript="hello namaste", intent="UNKNOWN", already_collected={}
Output: {"extracted":{},"all_collected":{},"still_needed":[],"status":"incomplete","hinglish_response":"Namaste! Aap kya karna chahte hain? Paisa bhejna, balance check, ya kuch aur?"}

Example 4 — HARD_REPROMPT (audio unclear):
Input: transcript="...", intent="UNKNOWN", audio_tier="HARD_REPROMPT", already_collected={}
Output: {"extracted":{},"all_collected":{},"still_needed":[],"status":"incomplete","hinglish_response":"Maafi, awaaz clearly nahi aayi. Thoda paas aakar dobara bolein?"}

Example 5 — FIR_THEFT, partial info:
Input: transcript="kal raat dukan mein ghuse teen log paanch hazaar le gaye", intent="FIR_THEFT", already_collected={}
Output: {"extracted":{"incident":"dukaan mein ghuse","num_perpetrators":"3","amount_stolen":"5000"},"all_collected":{"incident":"dukaan mein ghuse","num_perpetrators":"3","amount_stolen":"5000"},"still_needed":["time","location","victim_name"],"status":"incomplete","hinglish_response":"Samajh gaya. Yeh ghatna kab aur kahan hui? Aur aapka naam kya hai?"}"""


def new_session() -> dict:
    return {
        "intent":          None,
        "collected_slots": {},
        "verbatim":        [],
        "turns":           0,
        "eval": {
            "total_turns":    0,
            "hard_reprompts": 0,
            "soft_reprompts": 0,
            "accepts":        0,
            "avg_confidence": 0.0,
            "conf_samples":   [],
        }
    }


class SmartAgentDecisionLayer:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.store = SessionStore()
        print(f"Loading Smart Agent LLM ({model_id})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        try:
            torch.cuda.empty_cache()
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16
            ).to(DEVICE)
        except Exception as e:
            print(f"  [WARN] LLM GPU fetch failed ({e}). Using CPU float32.")
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float32
            ).to("cpu")

        self.llm_device = self.llm.device
        print(f"  [SESSION] LLM loaded on {self.llm_device}.")

    # ── LLM intent inference (fallback for UNKNOWN/uncertain cases) ──────────
    def _llm_infer_intent(self, transcript: str) -> str:
        """
        Ask the LLM to classify intent when DistilBERT returned UNKNOWN.
        Returns one of the supported intent labels or UNKNOWN.
        This is the smart fallback — the LLM understands context far better
        than a small 237-sample DistilBERT for ambiguous/OOD english inputs.
        """
        # Very short prompt — 0.5B models respond better to concise instructions
        user_msg = (
            f'User said: "{transcript}"\n\n'
            f"What is the intent? Pick EXACTLY ONE:\n"
            f"SEND_MONEY / CHECK_BALANCE / BILL_PAYMENT / RECEIVE_MONEY / EXPENSE_LOG / "
            f"FIR_THEFT / FIR_ASSAULT / ASSET_DECLARATION / HEALTH_RECORD / UNKNOWN\n\n"
            f"Respond with ONLY the intent label, nothing else."
        )
        messages = [
            {"role": "system", "content": "You are an intent classifier. Output ONLY the intent label."},
            {"role": "user",   "content": user_msg},
        ]
        text   = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.llm_device)
        with torch.no_grad():
            out = self.llm.generate(
                **inputs,
                max_new_tokens=8,          # just one label word
                do_sample=False,           # greedy — deterministic for classification
                pad_token_id=self.tokenizer.eos_token_id,
            )
        label = self.tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip().split()[0].upper()  # take first word, uppercase

        VALID_INTENTS = {
            "SEND_MONEY", "CHECK_BALANCE", "BILL_PAYMENT", "RECEIVE_MONEY",
            "EXPENSE_LOG", "FIR_THEFT", "FIR_ASSAULT", "ASSET_DECLARATION",
            "HEALTH_RECORD", "UNKNOWN",
        }
        return label if label in VALID_INTENTS else "UNKNOWN"

    # ── Format conversation history for the prompt ────────────────────────────
    def _format_history(self, session: dict) -> str:
        turns = session.get("verbatim", [])
        if not turns:
            return "  (first turn — no history yet)"
        # Only send last 6 turns to keep context short for 0.5B model
        return "\n".join(f"  {t}" for t in turns[-6:])

    # ── Core LLM call ─────────────────────────────────────────────────────────
    def _llm_reason(
        self,
        session:    dict,
        transcript: str,
        intent:     str,
        conf:       float,
        tier:       str,
    ) -> dict:
        collected = session.get("collected_slots", {})
        history   = self._format_history(session)

        user_msg = (
            f"CONVERSATION HISTORY:\n{history}\n\n"
            f"LATEST USER TURN:\n"
            f"  transcript   : \"{transcript}\"\n"
            f"  intent       : {intent}\n"
            f"  confidence   : {conf:.2f}\n"
            f"  audio_tier   : {tier}  (ACCEPT=clear, SOFT_REPROMPT=uncertain, HARD_REPROMPT=unclear)\n\n"
            f"ALREADY COLLECTED:\n"
            f"  {json.dumps(collected, ensure_ascii=False) if collected else '(nothing yet)'}\n\n"
            f"Extract new info, determine missing fields, respond in Hinglish. Return JSON only."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        text   = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.llm_device)

        with torch.no_grad():
            output_ids = self.llm.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=True,
                temperature=0.25,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        raw = self.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Parse JSON — be generous: find first {...} block
        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                # Validate keys exist
                data.setdefault("extracted",         {})
                data.setdefault("all_collected",      collected)
                data.setdefault("still_needed",       [])
                data.setdefault("status",             "incomplete")
                data.setdefault("hinglish_response",  "")
                return data
        except Exception:
            pass

        # Graceful fallback — don't crash the whole pipeline
        fallback_msg = (
            "Maafi chahta hoon, please thoda clearly dobara bolein."
            if tier == "HARD_REPROMPT"
            else "Theek hai. Aage bataiye — aapko kya karna hai?"
        )
        return {
            "extracted":         {},
            "all_collected":     collected,
            "still_needed":      [],
            "status":            "incomplete",
            "hinglish_response": fallback_msg,
        }

    # ── Eval tracking ─────────────────────────────────────────────────────────
    def _update_eval(self, session: dict, tier: str, conf: float):
        e = session["eval"]
        e["total_turns"] += 1
        if   tier == "ACCEPT":        e["accepts"]        += 1
        elif tier == "SOFT_REPROMPT": e["soft_reprompts"] += 1
        else:                         e["hard_reprompts"] += 1
        e["conf_samples"].append(conf)
        e["avg_confidence"] = round(sum(e["conf_samples"]) / len(e["conf_samples"]), 4)

    # ── Main entry point ──────────────────────────────────────────────────────
    def process_turn(self, session_id: str, pipeline_output: dict) -> dict:
        session  = self.store.get(session_id) or new_session()
        session["turns"] += 1

        transcript = pipeline_output.get("transcript", "")
        tier       = pipeline_output.get("status", "ACCEPT")
        intent     = (
            pipeline_output.get("intent_raw")
            or pipeline_output.get("intent", "UNKNOWN").split(" ")[0]
        )
        conf    = pipeline_output.get("confidence", 0.0)
        amount  = pipeline_output.get("amount", "UNKNOWN")
        latency = pipeline_output.get("latency", {})

        # Store verbatim record
        session["verbatim"].append(
            f"[T{session['turns']} | {tier} | conf={conf:.2f} | intent={intent}]: '{transcript}'"
        )
        self._update_eval(session, tier, conf)

        # Persist intent once detected by pipeline
        if not session["intent"] and intent not in ("UNKNOWN", "", None):
            session["intent"] = intent

        # ── LLM intent fallback ───────────────────────────────────────────────
        # If DistilBERT returned UNKNOWN (entropy OOD gate fired) AND we have
        # no intent from previous turns, try the LLM. It understands context
        # far better than the small classifier for mixed English/Hinglish input.
        if not session["intent"] and intent == "UNKNOWN":
            try:
                inferred = self._llm_infer_intent(transcript)
                if inferred != "UNKNOWN":
                    session["intent"] = inferred
                    # Add a note to the verbatim log
                    session["verbatim"][-1] += f" [LLM inferred: {inferred}]"
                    intent = inferred
            except Exception as e:
                print(f"  [WARN] LLM intent inference failed: {e}")

        # Pre-populate amount from pipeline if not yet collected
        if amount not in ("UNKNOWN", "", None) and "amount" not in session["collected_slots"]:
            session["collected_slots"]["amount"] = amount

        # ── Ask LLM to reason over the full conversation ──────────────────
        llm_out = self._llm_reason(
            session,
            transcript,
            session["intent"] or intent,
            conf,
            tier,
        )

        # Merge extractions — skip obvious junk values
        _junk = {"unknown", "none", "null", "", "n/a"}
        for src_dict in (llm_out.get("extracted", {}), llm_out.get("all_collected", {})):
            for k, v in src_dict.items():
                v_str = str(v).strip()
                if v_str.lower() not in _junk:
                    session["collected_slots"].setdefault(k, v_str)

        still_needed    = llm_out.get("still_needed", [])
        status          = llm_out.get("status", "incomplete")
        hinglish_resp   = llm_out.get("hinglish_response", "")

        if status == "complete":
            final_record = {
                "intent":      session["intent"],
                "total_turns": session["turns"],
            }
            final_record.update(session["collected_slots"])
            final_record["verbatim"] = session["verbatim"]
            eval_summary = session["eval"]
            self.store.delete(session_id)

            return {
                "status":        "complete",
                "message":       "All information collected. Structured record ready.",
                "final_record":  final_record,
                "eval_summary":  eval_summary,
                "agent_prompt":  hinglish_resp or "Shukriya! Aapki saari jankari save ho gayi.",
                "latency":       latency,
            }

        self.store.set(session_id, session)
        return {
            "status":          "incomplete",
            "intent":          session["intent"],
            "missing_slots":   still_needed,
            "collected_slots": session["collected_slots"],
            "agent_prompt":    hinglish_resp,
            "turn":            session["turns"],
            "latency":         latency,
            "eval":            session["eval"],
        }


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = SmartAgentDecisionLayer()

    def run(label, turns):
        print(f"\n{'='*55}\n  {label}\n{'='*55}")
        sid = f"test_{int(time.time())}"
        for i, t in enumerate(turns, 1):
            print(f"\n[Turn {i}] '{t['transcript']}'  conf={t['confidence']}  tier={t['status']}")
            r = agent.process_turn(sid, t)
            if r["status"] == "complete":
                print("✅  COMPLETE")
                print(json.dumps(r["final_record"], indent=2, ensure_ascii=False))
            else:
                print(f"🎙️  {r.get('agent_prompt', '')}")
                print(f"   missing={r.get('missing_slots', [])}  collected={r.get('collected_slots', {})}")

    run("Send Money", [
        {"status": "ACCEPT", "transcript": "1000 rupees Rahul ko send kar do",
         "intent_raw": "SEND_MONEY", "amount": "1000", "confidence": 0.72, "latency": {}},
    ])

    run("FIR Theft multi-turn", [
        {"status": "ACCEPT", "transcript": "kal raat dukan mein ghuse teen log paanch hazaar le gaye",
         "intent_raw": "UNKNOWN", "amount": "5000", "confidence": 0.65, "latency": {}},
        {"status": "ACCEPT", "transcript": "model town amritsar, raat das baje ka waqt tha",
         "intent_raw": "UNKNOWN", "amount": "UNKNOWN", "confidence": 0.72, "latency": {}},
        {"status": "ACCEPT", "transcript": "mera naam ramesh sharma hai",
         "intent_raw": "UNKNOWN", "amount": "UNKNOWN", "confidence": 0.80, "latency": {}},
    ])