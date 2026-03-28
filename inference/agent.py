"""
agent.py  —  SmartAgentDecisionLayer
--------------------------------------
Handles multi-turn Hinglish voice conversations.

Session persistence: Redis (with in-memory fallback if Redis unavailable).
Confidence tiers (set in pipeline.py):
  ACCEPT        → extract slots, advance conversation
  SOFT_REPROMPT → tentatively extract, ask confirmation before committing
  HARD_REPROMPT → reject turn, ask user to repeat clearly
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
    _redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    _redis_client.ping()
    REDIS_AVAILABLE = True
    print("  [SESSION] Redis connected — sessions persist across restarts.")
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
            raw = _redis_client.get(self._key(sid))
            return json.loads(raw) if raw else None
        return self._mem.get(sid)

    def set(self, sid: str, data: dict):
        if REDIS_AVAILABLE:
            _redis_client.setex(self._key(sid), SESSION_TTL, json.dumps(data, ensure_ascii=False))
        else:
            self._mem[sid] = data

    def delete(self, sid: str):
        if REDIS_AVAILABLE:
            _redis_client.delete(self._key(sid))
        else:
            self._mem.pop(sid, None)


# ── Goal schemas ──────────────────────────────────────────────────────────────
class GoalSchema:
    def __init__(self, name, required_slots, description, confirmation_prompt=None):
        self.name                = name
        self.required_slots      = required_slots
        self.description         = description
        self.confirmation_prompt = confirmation_prompt


GOAL_SCHEMAS = {
    "SEND_MONEY": GoalSchema(
        "SEND_MONEY", ["amount", "recipient"],
        "The user wants to transfer money to someone.",
        confirmation_prompt="Kya aap {amount} rupaye {recipient} ko bhejana chahte hain? Confirm karein.",
    ),
    "CHECK_BALANCE": GoalSchema(
        "CHECK_BALANCE", ["account_type"],
        "The user wants to check their bank balance.",
    ),
    "BILL_PAYMENT": GoalSchema(
        "BILL_PAYMENT", ["bill_type", "amount", "biller_name"],
        "The user wants to pay a utility bill.",
    ),
    "RECEIVE_MONEY": GoalSchema(
        "RECEIVE_MONEY", ["amount", "sender"],
        "The user wants to request money from someone.",
    ),
    "EXPENSE_LOG": GoalSchema(
        "EXPENSE_LOG", ["amount", "category"],
        "The user wants to log an expense.",
    ),
    "FIR_THEFT": GoalSchema(
        "FIR_THEFT",
        ["incident", "amount", "perpetrators", "time", "location", "victim_name"],
        "The user is reporting a theft or robbery to the police.",
    ),
    "FIR_ASSAULT": GoalSchema(
        "FIR_ASSAULT",
        ["incident", "time", "location", "victim_name", "accused_description"],
        "The user is reporting a physical assault to the police.",
    ),
    "ASSET_DECLARATION": GoalSchema(
        "ASSET_DECLARATION",
        ["asset_type", "size_or_value", "location", "beneficiary", "declarant_name"],
        "An elderly person is declaring how their assets should be distributed to family.",
    ),
    "HEALTH_RECORD": GoalSchema(
        "HEALTH_RECORD",
        ["child_age", "weight", "symptom", "household_id"],
        "An ASHA worker is recording a child health observation after a home visit.",
    ),
}

PIPELINE_INTENT_MAP = {
    "SEND_MONEY":    "SEND_MONEY",
    "CHECK_BALANCE": "CHECK_BALANCE",
    "BILL_PAYMENT":  "BILL_PAYMENT",
    "RECEIVE_MONEY": "RECEIVE_MONEY",
    "EXPENSE_LOG":   "EXPENSE_LOG",
}

EXTENDED_KEYWORDS = {
    "FIR_THEFT":        ["chori", "ghuse", "loot", "le gaye", "churaya", "theft", "robbery", "dakaiti"],
    "FIR_ASSAULT":      ["maara", "pita", "assault", "maar", "dhamki", "lathi", "knife", "chaku"],
    "ASSET_DECLARATION":["zameen", "property", "makaan", "wirasat", "dena hai", "bete ko", "beti ko",
                         "inheritance", "will", "vasiyat"],
    "HEALTH_RECORD":    ["baccha", "weight", "kilo", "bimaar", "bukhar", "khana nahi", "poshan",
                         "nutrition", "asha", "home visit"],
}


def detect_intent_from_transcript(transcript: str) -> str | None:
    t = transcript.lower()
    for intent, keywords in EXTENDED_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return intent
    return None


def new_session() -> dict:
    return {
        "intent":               None,
        "collected_slots":      {},
        "pending_slots":        {},
        "verbatim":             [],
        "turns":                0,
        "hard_reprompt_streak": 0,
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
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct"):
        self.store = SessionStore()
        print(f"Loading Smart Agent LLM ({model_id})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        try:
            torch.cuda.empty_cache()
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16
            ).to(DEVICE)
        except Exception as e:
            print(f"  [WARN] LLM GPU failed ({e}). Using CPU float32.")
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float32
            ).to("cpu")
        self.device = self.llm.device

    def _resolve_schema(self, session: dict, pipeline_intent: str, transcript: str) -> GoalSchema | None:
        if session["intent"]:
            return GOAL_SCHEMAS.get(session["intent"])
        # Extended use-cases detected by keyword (take priority — more specific)
        extended = detect_intent_from_transcript(transcript)
        if extended:
            session["intent"] = extended
            return GOAL_SCHEMAS[extended]
        # Financial intents from pipeline label
        mapped = PIPELINE_INTENT_MAP.get(pipeline_intent)
        if mapped:
            session["intent"] = mapped
            return GOAL_SCHEMAS[mapped]
        return None

    def _llm_extract(self, transcript: str, schema: GoalSchema, session: dict) -> dict:
        collected = session["collected_slots"]
        missing   = [s for s in schema.required_slots if s not in collected]

        sys_prompt = (
            f"You are a careful Hinglish voice assistant helping collect structured information.\n"
            f"Context: {schema.description}\n"
            f"Required slots still missing: {', '.join(missing) if missing else 'None'}.\n"
            f"Already confirmed: {json.dumps(collected, ensure_ascii=False)}\n"
            f"New user input: \"{transcript}\"\n\n"
            f"Rules:\n"
            f"1. Extract ONLY slots from the required list. Do not invent slot names.\n"
            f"2. If slots are still missing after extraction, write ONE short Hinglish question "
            f"for exactly ONE missing slot. Make it sound natural and conversational.\n"
            f"3. If ALL required slots are now filled, write exactly the word SUCCESS.\n\n"
            f"Return ONLY valid JSON, no explanation, no markdown:\n"
            f'{{\"extracted\": {{\"slot_name\": \"value\"}}, \"hinglish_question\": \"question or SUCCESS\"}}'
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": "Extract and respond with JSON only."}
        ]
        text   = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.2,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )

        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                data["extracted"] = {
                    k: v for k, v in data.get("extracted", {}).items()
                    if k in schema.required_slots
                }
                return data
        except Exception:
            pass

        fallback_q = (
            f"Mujhe {missing[0]} ke baare mein jankari chahiye. Kya aap bata sakte hain?"
            if missing else "SUCCESS"
        )
        return {"extracted": {}, "hinglish_question": fallback_q}

    def _update_eval(self, session: dict, tier: str, conf: float):
        e = session["eval"]
        e["total_turns"] += 1
        if tier == "ACCEPT":
            e["accepts"] += 1
        elif tier == "SOFT_REPROMPT":
            e["soft_reprompts"] += 1
        else:
            e["hard_reprompts"] += 1
        e["conf_samples"].append(conf)
        e["avg_confidence"] = round(sum(e["conf_samples"]) / len(e["conf_samples"]), 4)

    def process_turn(self, session_id: str, pipeline_output: dict) -> dict:
        session = self.store.get(session_id) or new_session()
        session["turns"] += 1

        transcript = pipeline_output.get("transcript", "")
        tier       = pipeline_output.get("status", "ACCEPT")
        # Use intent_raw (clean label) if present, fall back to parsing the intent string
        raw_intent = pipeline_output.get("intent_raw") or pipeline_output.get("intent", "UNKNOWN").split(" ")[0]
        amount     = pipeline_output.get("amount", "UNKNOWN")
        conf       = pipeline_output.get("confidence", 0.0)
        raw_conf   = pipeline_output.get("raw_confidence", conf)
        kw_flag    = pipeline_output.get("keyword_override", False)
        latency    = pipeline_output.get("latency", {})

        # Always preserve verbatim record
        session["verbatim"].append(
            f"[T{session['turns']} | {tier}{' KW' if kw_flag else ''} | conf={conf:.2f}]: '{transcript}'"
        )
        self._update_eval(session, tier, conf)

        # ── HARD_REPROMPT ─────────────────────────────────────────────────
        if tier == "HARD_REPROMPT":
            session["hard_reprompt_streak"] += 1
            self.store.set(session_id, session)

            streak = session["hard_reprompt_streak"]
            if streak >= 3:
                msg = (
                    "Mujhe aapki awaaz bilkul samajh nahi aa rahi. "
                    "Kya aap thoda aur paas aa ke clearly bol sakte hain? "
                    "Ya text mein type karein."
                )
            elif streak == 2:
                msg = "Phir se clearly nahi samajh aaya. Kya aap dhire aur saaf bol sakte hain?"
            else:
                msg = "Maafi chahta/chahti hoon, clearly samajh nahi aaya. Kya aap dobara bol sakte hain?"

            return {
                "status":       "hard_reprompt",
                "tier":         tier,
                "confidence":   conf,
                "streak":       streak,
                "agent_prompt": msg,
                "latency":      latency,
                "eval":         session["eval"],
            }

        # Good turn — reset streak
        session["hard_reprompt_streak"] = 0

        # ── Resolve schema ────────────────────────────────────────────────
        schema = self._resolve_schema(session, raw_intent, transcript)

        if schema is None:
            self.store.set(session_id, session)
            return {
                "status": "intent_unclear",
                "agent_prompt": (
                    "Namaste! Kya aap kisi crime ki report karna chahte hain, "
                    "property ke baare mein kuch kehna chahte hain, "
                    "ya bacche ki health report darz karni hai?"
                ),
                "latency": latency,
                "eval":    session["eval"],
            }

        # Pre-populate amount if pipeline extracted it
        if amount != "UNKNOWN" and "amount" in schema.required_slots:
            session["collected_slots"].setdefault("amount", amount)

        # ── SOFT_REPROMPT: extract tentatively, seek confirmation ─────────
        if tier == "SOFT_REPROMPT":
            llm_out = self._llm_extract(transcript, schema, session)
            pending = {k: v for k, v in llm_out.get("extracted", {}).items()}

            if pending:
                session["pending_slots"] = pending
                items       = ", ".join(f"{k}: {v}" for k, v in pending.items())
                confirm_msg = (
                    f"Mujhe lagta hai aapne kaha: {items}. "
                    "Kya yeh sahi hai? Haan ya na bolein."
                )
            else:
                confirm_msg = (
                    "Thodi awaaz clear nahi thi. "
                    "Kya aap dobara thoda clearly bol sakte hain?"
                )

            self.store.set(session_id, session)
            return {
                "status":        "soft_reprompt",
                "tier":          tier,
                "confidence":    conf,
                "pending_slots": pending,
                "agent_prompt":  confirm_msg,
                "latency":       latency,
                "eval":          session["eval"],
            }

        # ── ACCEPT ────────────────────────────────────────────────────────
        # Check if this turn is confirming/rejecting pending soft slots
        if session.get("pending_slots"):
            t_lower = transcript.lower()
            if any(w in t_lower for w in ["haan", "yes", "sahi", "correct", "bilkul", "theek"]):
                session["collected_slots"].update(session["pending_slots"])
                session["pending_slots"] = {}
            elif any(w in t_lower for w in ["na", "nahi", "no", "galat", "wrong"]):
                session["pending_slots"] = {}
                missing = [s for s in schema.required_slots if s not in session["collected_slots"]]
                self.store.set(session_id, session)
                return {
                    "status":          "incomplete",
                    "intent":          session["intent"],
                    "missing_slots":   missing,
                    "collected_slots": session["collected_slots"],
                    "agent_prompt": (
                        f"Koi baat nahi. Kya aap {missing[0]} ke baare mein dobara bata sakte hain?"
                        if missing else "Theek hai, aage batayein."
                    ),
                    "latency": latency,
                    "eval":    session["eval"],
                }
            # Neither yes nor no — fall through and treat as new input

        # Standard slot extraction
        llm_out = self._llm_extract(transcript, schema, session)
        for k, v in llm_out.get("extracted", {}).items():
            session["collected_slots"][k] = str(v)

        missing_slots = [s for s in schema.required_slots if s not in session["collected_slots"]]
        agent_msg     = llm_out.get("hinglish_question", "")
        all_done      = len(missing_slots) == 0

        if all_done:
            final_record = {
                "intent":      session["intent"],
                "total_turns": session["turns"],
            }
            final_record.update(session["collected_slots"])
            final_record["verbatim"] = session["verbatim"]

            eval_summary = session["eval"]
            self.store.delete(session_id)

            return {
                "status":       "complete",
                "message":      "All fields collected. Structured record ready.",
                "final_record": final_record,
                "eval_summary": eval_summary,
                "latency":      latency,
            }

        self.store.set(session_id, session)
        return {
            "status":          "incomplete",
            "intent":          session["intent"],
            "missing_slots":   missing_slots,
            "collected_slots": session["collected_slots"],
            "agent_prompt":    agent_msg,
            "turn":            session["turns"],
            "latency":         latency,
            "eval":            session["eval"],
        }


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = SmartAgentDecisionLayer()

    def run(label, turns):
        print(f"\n{'='*55}\n  {label}\n{'='*55}")
        sid = f"test_{label.lower().replace(' ', '_')}"
        for i, t in enumerate(turns, 1):
            print(f"\n[Turn {i}] '{t['transcript']}'  conf={t['confidence']}  tier={t['status']}")
            r = agent.process_turn(sid, t)
            if r["status"] == "complete":
                print("✅  COMPLETE")
                print(json.dumps(r["final_record"], indent=2, ensure_ascii=False))
                print(f"   Eval: {r['eval_summary']}")
            elif r["status"] == "hard_reprompt":
                print(f"🔇 HARD [{r['streak']}]: {r['agent_prompt']}")
            elif r["status"] == "soft_reprompt":
                print(f"🟡 SOFT: {r['agent_prompt']}")
            else:
                print(f"🎙️  {r.get('agent_prompt', '')}")
                print(f"   status={r['status']}  missing={r.get('missing_slots', '')}")

    run("Financial — previously failing (send with verb)", [
        {"status": "ACCEPT", "transcript": "1000 rupees Rahul ko send kar do",
         "intent_raw": "SEND_MONEY", "amount": "1000", "confidence": 0.70,
         "keyword_override": True, "latency": {}},
    ])

    run("FIR Theft with soft reprompt", [
        {"status": "ACCEPT",        "transcript": "kal raat mere dukan mein ghuse teen log paanch hazaar le gaye",
         "intent_raw": "UNKNOWN",   "amount": "5000", "confidence": 0.78, "keyword_override": False, "latency": {}},
        {"status": "SOFT_REPROMPT", "transcript": "model town amritsar sharma store",
         "intent_raw": "UNKNOWN",   "amount": "UNKNOWN", "confidence": 0.48, "keyword_override": False, "latency": {}},
        {"status": "ACCEPT",        "transcript": "haan sahi hai",
         "intent_raw": "UNKNOWN",   "amount": "UNKNOWN", "confidence": 0.81, "keyword_override": False, "latency": {}},
        {"status": "ACCEPT",        "transcript": "mera naam ramesh sharma hai raat das baje ka waqt tha",
         "intent_raw": "UNKNOWN",   "amount": "UNKNOWN", "confidence": 0.77, "keyword_override": False, "latency": {}},
    ])

    run("ASHA Health Record", [
        {"status": "ACCEPT", "transcript": "teen saal ka baccha weight barah kilo khana nahi khata",
         "intent_raw": "UNKNOWN", "amount": "UNKNOWN", "confidence": 0.79, "keyword_override": False, "latency": {}},
        {"status": "ACCEPT", "transcript": "ghar number 47 ward 3",
         "intent_raw": "UNKNOWN", "amount": "UNKNOWN", "confidence": 0.83, "keyword_override": False, "latency": {}},
    ])

    run("Hard reprompt escalation", [
        {"status": "HARD_REPROMPT", "transcript": "...", "intent_raw": "UNKNOWN",
         "amount": "UNKNOWN", "confidence": 0.22, "keyword_override": False, "latency": {}},
        {"status": "HARD_REPROMPT", "transcript": "...", "intent_raw": "UNKNOWN",
         "amount": "UNKNOWN", "confidence": 0.19, "keyword_override": False, "latency": {}},
        {"status": "HARD_REPROMPT", "transcript": "...", "intent_raw": "UNKNOWN",
         "amount": "UNKNOWN", "confidence": 0.17, "keyword_override": False, "latency": {}},
        {"status": "ACCEPT",        "transcript": "chori ho gayi das hazaar le gaye",
         "intent_raw": "UNKNOWN",   "amount": "10000", "confidence": 0.76, "keyword_override": False, "latency": {}},
    ])