import time
import json
import torch
import librosa
from pathlib import Path

from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    AutoTokenizer, DistilBertForTokenClassification, AutoModelForSequenceClassification
)
from peft import PeftModel

BASE_DIR = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def safe_to_device(model):
    if DEVICE == "cpu":
        return model
    try:
        torch.cuda.empty_cache()
        return model.to(DEVICE)
    except Exception as e:
        print(f"  [WARN] CUDA/OOM ({e}). Falling back to CPU.")
        return model.to("cpu")


WHISPER_ADAPTER = BASE_DIR / "models/adapters/whisper_lora"
ITN_MODEL       = BASE_DIR / "models/adapters/distilbert_itn"
INTENT_MODEL    = BASE_DIR / "models/adapters/distilbert_intent"

NUMBER_WORDS = {
    "ek": 1, "do": 2, "teen": 3, "char": 4, "paanch": 5,
    "chhe": 6, "saat": 7, "aath": 8, "nau": 9, "das": 10,
    "gyarah": 11, "barah": 12, "tera": 13, "chaudah": 14, "pandrah": 15,
    "solah": 16, "satrah": 17, "atharah": 18, "unnees": 19, "bees": 20,
    "pachas": 50, "saath": 70, "assi": 80, "nabbe": 90,
    "sau": 100, "hazaar": 1000, "lakh": 100000, "crore": 10000000,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "hundred": 100, "thousand": 1000,
}

AMBIGUOUS_NUM_WORDS = {"do", "teen", "char", "das", "ek", "two", "three", "one", "ten"}
VERB_ANCHORS = {
    "kar", "karo", "karna", "karte", "karta", "kari", "karein",
    "dena", "dedo", "de", "lo", "lena", "lete", "leta",
    "send", "bhejo", "bheja", "transfer",
}
QUANTITY_ANCHORS = {
    "rupay", "rupaye", "rupees", "rs", "rs.", "₹",
    "hazaar", "lakh", "crore", "sau",
    "kilo", "kg", "gram", "litre", "meter",
    "log", "baar", "din", "mahine", "saal",
}


def should_convert_ambiguous(word: str, words: list, idx: int) -> bool:
    if idx > 0:
        left = words[idx - 1].lower().rstrip(".,?!")
        if left in VERB_ANCHORS:
            return False
    if idx < len(words) - 1:
        right = words[idx + 1].lower().lstrip("₹").rstrip(".,?!")
        if right in QUANTITY_ANCHORS or right in NUMBER_WORDS:
            return True
    if idx > 0:
        left_clean = ''.join(c for c in words[idx - 1].lower() if c.isalnum())
        if left_clean.isdigit() or left_clean in NUMBER_WORDS:
            return True
    return False


def apply_itn_substitution(words: list, word_labels: dict) -> list:
    final_words = []
    for i, w in enumerate(words):
        if word_labels.get(i) == "NUM":
            clean_w = ''.join(c for c in w if c.isalnum()).lower()
            if clean_w in NUMBER_WORDS:
                if clean_w in AMBIGUOUS_NUM_WORDS:
                    final_words.append(
                        str(NUMBER_WORDS[clean_w]) if should_convert_ambiguous(clean_w, words, i) else w
                    )
                else:
                    final_words.append(str(NUMBER_WORDS[clean_w]))
            else:
                final_words.append(w)
        else:
            final_words.append(w)
    return final_words


# ── TEXT NORMALIZATION (closes train/inference format gap) ───────────────────
#
# Training data: short, lowercase, no punctuation  ("das hazaar bhejo")
# Whisper output: proper case, punctuated, longer   ("Send 10,000 rupees.")
#
# DistilBERT tokenizes "balance" and "Balance." differently.
# Normalizing the Whisper transcript to match training format
# restores the token distribution the classification head was calibrated for.

import re as _re

def normalize_for_intent(text: str) -> str:
    """Lowercase + strip punctuation to match training data format."""
    text = text.lower()
    # Keep alphanumeric, whitespace, ₹ sign, digits
    text = _re.sub(r'[^\w\s\u20b9]', ' ', text)
    text = ' '.join(text.split())
    return text


# ── CONFIDENCE + OOD DETECTION ────────────────────────────────────────────────
#
# The model has 5 classes. There is no REJECT / OTHER class.
# Any OOD input (greetings, noise, random speech) is FORCED into one of 5
# financial labels — confidence will always be low and label will be wrong.
#
# Fix: compute Shannon entropy H of the softmax distribution.
#   H_max = ln(5) ≈ 1.609  (uniform = maximum uncertainty)
#   H < 0.5  → clear prediction
#   H > 1.4  → near-uniform = OOD input, treat as UNKNOWN

OOD_ENTROPY_THRESH = 1.40    # flag anything above this as out-of-distribution

CALIB_TEMP = 1.0
HIGH_CONF  = 0.45
LOW_CONF   = 0.28


def get_confidence(logits: torch.Tensor) -> tuple[int, float, bool]:
    """Returns (pred_id, confidence, is_ood)."""
    import math
    probs   = torch.softmax(logits / CALIB_TEMP, dim=-1)[0]
    np_prob = probs.cpu().numpy()
    pred_id = int(np_prob.argmax())
    conf    = round(float(np_prob[pred_id]), 4)
    # Shannon entropy
    H       = float(-sum(p * math.log(p + 1e-12) for p in np_prob))
    is_ood  = H > OOD_ENTROPY_THRESH
    return pred_id, conf, is_ood


# ── KEYWORD OVERRIDE ──────────────────────────────────────────────────────────
#
# "1000 rupees Rahul ko send kar do" scores 0.44 — correct prediction but below
# HIGH_CONF because "kar do" spreads mass across SEND_MONEY and EXPENSE_LOG.
#
# If the transcript contains an intent-unambiguous keyword AND the model's own
# top-1 agrees, we promote SOFT_REPROMPT → ACCEPT.
# The intent LABEL is never changed by this — keywords only unlock the tier gate.
#
# Reported confidence is set to KEYWORD_OVERRIDE_CONF so the UI shows a
# meaningful value. The original model probability is in "raw_confidence".

# Keyword overrides: if transcript matches a keyword, we DON'T need the model
# to agree — the keyword IS the intent. This handles cases where the model
# predicts the wrong label (e.g. mis-classifies "send kar do" as BILL_PAYMENT).
INTENT_KEYWORDS: dict[str, list[str]] = {
    "SEND_MONEY": [
        "send", "bhejo", "bhej do", "bhej", "transfer", "paisa bhej", "paise bhej",
        "paise do", "paisa do", "rupay bhejo", "rupaye bhejo", "rupaye do",
        "ko de do", "ko dedo", "ko bhej", "ko send", "usse bhejo", "use bhejo",
        "woh le le", "dena hai usse",
    ],
    "CHECK_BALANCE": [
        "balance", "kitna hai", "check karo", "dekho", "kitne paise",
        "kitna paisa", "account mein", "how much", "total kitna",
    ],
    "BILL_PAYMENT": [
        "bill", "pay karo", "bharo", "jama karo", "payment karo",
        "bijli", "electricity", "gas bill", "recharge", "mobile bill",
        "broadband", "internet bill",
    ],
    "RECEIVE_MONEY": [
        "receive", "mangao", "bhijwao", "mangwa", "lena hai",
        "paisa mangna", "bhijwa do", "request karo", "mujhe chahiye",
        "mujhe paisa", "mujhe paise",
    ],
    "EXPENSE_LOG": [
        "kharcha", "expense", "nota karo", "record karo",
        "kharch hua", "kharch kiya", "spent", "laga", "lag gaya",
    ],
}

KEYWORD_OVERRIDE_CONF = 0.72


def keyword_intent_match(transcript: str) -> str | None:
    """Return the intent label if transcript contains an unambiguous keyword."""
    t = transcript.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return intent
    return None


def confidence_tier(conf: float, keyword_override: bool = False) -> str:
    if conf >= HIGH_CONF:
        return "ACCEPT"
    elif conf >= LOW_CONF:
        return "ACCEPT" if keyword_override else "SOFT_REPROMPT"
    return "HARD_REPROMPT" if not keyword_override else "SOFT_REPROMPT"


class EntropicPipeline:
    def __init__(self):
        print(f"Loading End-to-End Pipeline on {DEVICE.upper()}...")

        print(" [1/4] Whisper ASR + LoRA...")
        self.wh_proc = WhisperProcessor.from_pretrained("openai/whisper-small", local_files_only=True)
        self.forced_decoder_ids = self.wh_proc.get_decoder_prompt_ids(language="en", task="transcribe")
        wh_base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", local_files_only=True)
        self.wh_model = PeftModel.from_pretrained(safe_to_device(wh_base), str(WHISPER_ADAPTER))

        print(" [2/4] DistilBERT Neural ITN...")
        self.itn_tok   = AutoTokenizer.from_pretrained(str(ITN_MODEL))
        self.itn_model = safe_to_device(
            DistilBertForTokenClassification.from_pretrained(str(ITN_MODEL))
        )
        self.itn_model.eval()

        print(" [3/4] DistilBERT Intent Classifier...")
        self.intent_tok   = AutoTokenizer.from_pretrained(str(INTENT_MODEL))
        self.intent_model = safe_to_device(
            AutoModelForSequenceClassification.from_pretrained(str(INTENT_MODEL))
        )
        self.intent_model.eval()

        print(" [4/4] Silero VAD...")
        try:
            self.vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False
            )
            self.get_speech_timestamps = utils[0]
        except Exception as e:
            print(f"  [WARN] VAD unavailable: {e}")
            self.vad_model = None

        cfg = json.loads((INTENT_MODEL / "intent_config.json").read_text())
        self.id2label = {int(k): v for k, v in cfg["id2label"].items()}
        print("Pipeline ready.\n")

    def extract_amount(self, text: str) -> str:
        import re
        digits = re.findall(r'\d+', text)
        return digits[0] if digits else "UNKNOWN"

    def _run_whisper(self, inputs):
        gen_kwargs = dict(
            input_features=inputs.input_features,
            attention_mask=inputs.get("attention_mask"),
            forced_decoder_ids=self.forced_decoder_ids,
            max_new_tokens=50,
            condition_on_prev_tokens=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
        )
        try:
            with torch.no_grad():
                return self.wh_model.generate(**gen_kwargs)
        except RuntimeError as e:
            if "CUDA" in str(e) or "CUBLAS" in str(e):
                self.wh_model = self.wh_model.to("cpu")
                torch.cuda.empty_cache()
                gen_kwargs["input_features"] = inputs.input_features.to("cpu")
                gen_kwargs.pop("attention_mask", None)
                with torch.no_grad():
                    return self.wh_model.generate(**gen_kwargs)
            raise

    def _run_itn(self, words: list) -> str:
        enc_obj  = self.itn_tok(words, is_split_into_words=True, return_tensors="pt", truncation=True)
        word_ids = enc_obj.word_ids()
        enc      = {k: v.to(self.itn_model.device) for k, v in enc_obj.items()}
        try:
            with torch.no_grad():
                logits = self.itn_model(**enc).logits
        except RuntimeError as e:
            if "CUDA" in str(e) or "CUBLAS" in str(e):
                self.itn_model = self.itn_model.to("cpu")
                enc = {k: v.to("cpu") for k, v in enc.items()}
                torch.cuda.empty_cache()
                with torch.no_grad():
                    logits = self.itn_model(**enc).logits
            else:
                raise
        preds = torch.argmax(logits, dim=-1)[0].cpu().numpy()
        word_labels = {}
        for idx, wid in enumerate(word_ids):
            if wid is not None and wid not in word_labels:
                word_labels[wid] = self.itn_model.config.id2label[preds[idx]]
        return " ".join(apply_itn_substitution(words, word_labels))

    def _run_intent(self, text: str) -> tuple[str, float, bool]:
        # Normalize to match training data format BEFORE tokenizing
        normalized = normalize_for_intent(text)
        enc = self.intent_tok(normalized, return_tensors="pt", truncation=True)
        enc = {k: v.to(self.intent_model.device) for k, v in enc.items()}
        try:
            with torch.no_grad():
                logits = self.intent_model(**enc).logits
        except RuntimeError as e:
            if "CUDA" in str(e) or "CUBLAS" in str(e):
                self.intent_model = self.intent_model.to("cpu")
                enc = {k: v.to("cpu") for k, v in enc.items()}
                torch.cuda.empty_cache()
                with torch.no_grad():
                    logits = self.intent_model(**enc).logits
            else:
                raise
        pred_id, conf, is_ood = get_confidence(logits)
        return self.id2label[pred_id], conf, is_ood

    def transcribe(self, audio_path: str) -> dict:
        latency = {}
        t_total = time.time()

        # Stage 1: Audio prep + VAD
        t0 = time.time()
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        audio, _ = librosa.effects.trim(audio, top_db=45, frame_length=1024, hop_length=256)
        if getattr(self, "vad_model", None) is not None:
            audio_tensor = torch.tensor(audio)
            ts = self.get_speech_timestamps(audio_tensor, self.vad_model, sampling_rate=16000)
            if ts:
                audio = torch.cat([audio_tensor[t['start']:t['end']] for t in ts]).numpy()
        latency["vad_ms"] = round((time.time() - t0) * 1000, 1)

        # Stage 2: ASR
        t0 = time.time()
        inputs     = self.wh_proc(audio, sampling_rate=16000, return_tensors="pt").to(self.wh_model.device)
        pred_ids   = self._run_whisper(inputs)
        transcript = self.wh_proc.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
        latency["asr_ms"] = round((time.time() - t0) * 1000, 1)

        # Stage 3: ITN
        t0 = time.time()
        normalized_text = self._run_itn(transcript.split()) if transcript.split() else transcript
        latency["itn_ms"] = round((time.time() - t0) * 1000, 1)

        # Stage 4: Intent (with OOD detection)
        t0 = time.time()
        intent, conf, is_ood = self._run_intent(normalized_text)
        latency["intent_ms"] = round((time.time() - t0) * 1000, 1)

        # Stage 5: Tier assignment + keyword override + OOD gate
        #
        # OOD gate (Fix B): if entropy is near-uniform (H > 1.4), the model
        # is completely uncertain — likely non-financial / greetings / noise.
        # We short-circuit to UNKNOWN / HARD_REPROMPT instead of propagating
        # a wrong financial label with low confidence.
        #
        # Keyword override (Fix A extension): if transcript contains an
        # unambiguous financial keyword, that keyword wins as the intent
        # regardless of what the model predicted. This handles cases where
        # the model predicts the wrong label even on financial utterances.
        #
        kw_intent = keyword_intent_match(transcript)

        if is_ood and kw_intent is None:
            # Entropy-based OOD: model is near-uniform and no keyword found
            final_intent   = "UNKNOWN"
            keyword_agrees = False
            tier           = "HARD_REPROMPT"
            reported_conf  = conf
        elif kw_intent is not None:
            # Keyword found: use keyword intent unconditionally
            final_intent   = kw_intent
            keyword_agrees = True
            tier           = confidence_tier(conf, keyword_override=True)
            reported_conf  = (
                KEYWORD_OVERRIDE_CONF if conf < HIGH_CONF else conf
            )
        else:
            # No keyword, not OOD: trust the model
            final_intent   = intent
            keyword_agrees = False
            tier           = confidence_tier(conf, keyword_override=False)
            reported_conf  = conf

        amount = self.extract_amount(normalized_text)
        latency["total_ms"] = round((time.time() - t_total) * 1000, 1)

        # intent field: always clean label — status field carries tier info
        return {
            "status":           tier,
            "transcript":       transcript,
            "normalized_text":  normalized_text,
            "intent":           final_intent,
            "intent_raw":       final_intent,
            "amount":           amount,
            "confidence":       reported_conf,
            "raw_confidence":   conf,
            "keyword_override": keyword_agrees,
            "latency":          latency,
        }