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
from indic_transliteration import sanscript

BASE_DIR = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def safe_to_device(model):
    if DEVICE == "cpu":
        return model
    try:
        torch.cuda.empty_cache()
        return model.to(DEVICE)
    except Exception as e:
        print(f"  [WARN] CUDA NVML or OOM error ({e}). Falling back to CPU for this model.")
        return model.to("cpu")


# Load endpoints
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

class EntropicPipeline:
    def __init__(self):
        print(f"Loading End-to-End Pipeline on {DEVICE.upper()}...")
        
        print(" [1/3] Whisper ASR + LoRA...")
        self.wh_proc = WhisperProcessor.from_pretrained("openai/whisper-small", local_files_only=True)
        self.forced_decoder_ids = self.wh_proc.get_decoder_prompt_ids(language="hi", task="transcribe")
        wh_base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", local_files_only=True)
        wh_base = safe_to_device(wh_base)
        self.wh_model = PeftModel.from_pretrained(wh_base, str(WHISPER_ADAPTER))

        print(" [2/3] DistilBERT Neural ITN...")
        self.itn_tok = AutoTokenizer.from_pretrained(str(ITN_MODEL))
        itn_base = DistilBertForTokenClassification.from_pretrained(str(ITN_MODEL))
        self.itn_model = safe_to_device(itn_base)
        self.itn_model.eval()

        print(" [3/4] DistilBERT Intent Classifier...")
        self.intent_tok = AutoTokenizer.from_pretrained(str(INTENT_MODEL))
        intent_base = AutoModelForSequenceClassification.from_pretrained(str(INTENT_MODEL))
        self.intent_model = safe_to_device(intent_base)
        self.intent_model.eval()

        print(" [4/4] Silero VAD (Voice Activity Detection)...")
        try:
            self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
            self.get_speech_timestamps = utils[0]
        except Exception as e:
            print(f"  [WARN] Failed to pull Silero VAD from Github (Networking?): {e}")
            self.vad_model = None

        # Load Intent Config for thresholds and labels
        conf = json.loads((INTENT_MODEL / "intent_config.json").read_text())
        self.id2label = {int(k): v for k, v in conf["id2label"].items()}
        self.conf_thresh = conf.get("confidence_threshold", 0.60)
        
        print("Pipeline is extremely ready.\n")

    def extract_amount(self, text: str) -> str:
        import re
        digits = re.findall(r'\d+', text)
        return digits[0] if digits else "UNKNOWN"

    def transcribe(self, audio_path: str):
        t0 = time.time()
        
        # ==========================================
        # 1. ASR (Acoustic -> Roman Hinglish)
        # ==========================================
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Apply Neural VAD (Silero) if successfully loaded
        if getattr(self, "vad_model", None) is not None:
            audio_tensor = torch.tensor(audio)
            timestamps = self.get_speech_timestamps(audio_tensor, self.vad_model, sampling_rate=16000)
            if timestamps:
                # Reconstruct speech parts only
                audio = torch.cat([audio_tensor[ts['start']:ts['end']] for ts in timestamps]).numpy()

        inputs = self.wh_proc(audio, sampling_rate=16000, return_tensors="pt").to(self.wh_model.device)
        try:
            with torch.no_grad():
                pred_ids = self.wh_model.generate(
                    input_features=inputs.input_features, 
                    attention_mask=inputs.get("attention_mask"),
                    forced_decoder_ids=self.forced_decoder_ids,
                    max_new_tokens=50
                ) 
        except RuntimeError as e:
            if "CUDA" in str(e) or "CUBLAS" in str(e):
                print(f"  [WARN] CUDA crash during Whisper generation ({e}). Forcing CPU fallback.")
                self.wh_model = self.wh_model.to("cpu")
                inputs = inputs.to("cpu")
                with torch.no_grad():
                    pred_ids = self.wh_model.generate(
                        input_features=inputs.input_features, 
                        attention_mask=inputs.get("attention_mask"),
                        forced_decoder_ids=self.forced_decoder_ids,
                        max_new_tokens=50
                    )
            else:
                raise e
        transcript = self.wh_proc.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

        # Safety Fallback: if Whisper heavily forced Devanagari, cleanly transliterate back to Roman
        if any('\u0900' <= c <= '\u097f' for c in transcript):
            transcript = sanscript.transliterate(transcript, sanscript.DEVANAGARI, sanscript.ITRANS).lower()

        # ==========================================
        # 2. ITN (Text -> Normalized Numbers)
        # ==========================================
        words = transcript.lower().split()
        enc_itn_obj = self.itn_tok(words, is_split_into_words=True, return_tensors="pt", truncation=True)
        word_ids = enc_itn_obj.word_ids()
        
        enc_itn = {k: v.to(self.itn_model.device) for k, v in enc_itn_obj.items()}
        
        with torch.no_grad():
            itn_logits = self.itn_model(**enc_itn).logits
        itn_preds = torch.argmax(itn_logits, dim=-1)[0].cpu().numpy()

        word_labels = {}
        for idx, wid in enumerate(word_ids):
            if wid is not None and wid not in word_labels:
                word_labels[wid] = self.itn_model.config.id2label[itn_preds[idx]]

        final_words = []
        for i, w in enumerate(words):
            if word_labels.get(i) == "NUM":
                clean_w = ''.join(c for c in w if c.isalnum())
                if clean_w in NUMBER_WORDS:
                    final_words.append(str(NUMBER_WORDS[clean_w]))
                else:
                    final_words.append(w)
            else:
                final_words.append(w)
                
        normalized_text = " ".join(final_words)

        # ==========================================
        # 3. Intent Classification
        # ==========================================
        enc_int = self.intent_tok(normalized_text, return_tensors="pt", truncation=True)
        enc_int = {k: v.to(self.intent_model.device) for k, v in enc_int.items()}
        
        with torch.no_grad():
            int_logits = self.intent_model(**enc_int).logits
        probs = torch.softmax(int_logits, dim=-1)[0]
        max_prob, pred_id = probs.max(dim=0)
        
        intent = self.id2label[pred_id.item()]
        conf = max_prob.item()

        # ==========================================
        # 4. Amount Extraction & Confidence Gating
        # ==========================================
        amount = self.extract_amount(normalized_text)

        if conf < self.conf_thresh:
            status = "REPROMPT"
            intent = f"{intent} (Rejected: low confidence)"
        else:
            status = "SUCCESS"

        latency = round((time.time() - t0) * 1000, 2)

        return {
            "status": status,
            "transcript": transcript,
            "normalized_text": normalized_text,
            "intent": intent,
            "amount": amount,
            "confidence": round(conf, 4),
            "latency_ms": latency
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to the .wav file to test")
    args = parser.parse_args()

    pipeline = EntropicPipeline()
    res = pipeline.transcribe(args.audio)
    
    print("\n" + "="*50)
    print("🚀 ENTROPIC ASR PREDICTION")
    print("="*50)
    print(json.dumps(res, indent=2))
    print("="*50 + "\n")
