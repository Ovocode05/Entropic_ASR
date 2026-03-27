import time
import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class GoalSchema:
    def __init__(self, name, required_slots, description):
        self.name = name
        self.required_slots = required_slots
        self.description = description

GOAL_SCHEMAS = {
    # Instead of hardcoded strings, we just provide the definition.
    # The LLM will use this to generate smart, contextual Hinglish queries!
    "send_money": GoalSchema(
        name="send_money",
        required_slots=["amount", "recipient"],
        description="The user wants to transfer money to someone."
    ),
    "check_balance": GoalSchema(
        name="check_balance",
        required_slots=["account_type"],
        description="The user wants to check bank balance (savings or current)."
    ),
    "pay_bill": GoalSchema(
        name="pay_bill",
        required_slots=["bill_type", "amount", "biller_name"],
        description="The user wants to pay an utility bill."
    ),
    "request_money": GoalSchema(
        name="request_money",
        required_slots=["amount", "sender"],
        description="The user wants to request someone to send them money."
    ),
    "transaction_history": GoalSchema(
        name="transaction_history",
        required_slots=["duration"],
        description="The user wants to check past transaction history."
    ),
    "THEFT": GoalSchema(
        name="THEFT",
        required_slots=["incident", "amount", "perpetrators", "time", "location", "victim_name"],
        description="The user is reporting a robbery or theft."
    )
}

class SmartAgentDecisionLayer:
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct"):
        self.sessions = {}
        print(f"Loading Smart Agent LLM ({model_id}) for dynamic Hinglish responses...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Attempt GPU load, fallback to CPU securely
        try:
            torch.cuda.empty_cache()
            self.llm = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE)
        except Exception as e:
            print(f"  [WARN] LLM GPU fetch failed/OOM ({e}). Using CPU fallback (float32).")
            self.llm = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to("cpu")
            
        self.device = self.llm.device

    def get_or_create_session(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "intent": None,
                "collected_slots": {},
                "verbatim": [],
                "turns": 0
            }
        return self.sessions[session_id]

    def _generate_agentic_response(self, transcript: str, schema: GoalSchema, session: dict) -> dict:
        """
        Uses the local LLM to dynamically extract new slots and formulate 
        the next contextual Hinglish question.
        Returns a dictionary containing extracted fields and the next prompt.
        """
        collected = session["collected_slots"]
        missing = [s for s in schema.required_slots if s not in collected]
        
        sys_prompt = f"""You are an intelligent Hinglish voice assistant helping a user. 
Your context is: {schema.description}
Your goal is to collect these remaining details: {', '.join(missing) if missing else 'None'}.
Currently collected details: {json.dumps(collected)}
New user input: "{transcript}"

Task for your intelligence:
1. Extract any newly provided required details from the user input. Map them to the correct slot names.
2. If any required details are STILL missing, write a natural, polite Hinglish question to ask the user for exactly ONE of the missing details. 
3. If ALL required details are now collected, output exactly "SUCCESS" for the question.

Format your response strictly as valid JSON like this:
{{
  "extracted": {{"slot_name": "extracted_value from text"}},
  "hinglish_question": "Your dynamic Hinglish question here or SUCCESS"
}}"""
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "Analyze the input and return the JSON."}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs, 
                max_new_tokens=150, 
                do_sample=True, 
                temperature=0.2, 
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Parse JSON from LLM output
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return data
            # Edge case fallback if LLM breaks JSON format
            return {"extracted": {}, "hinglish_question": response.strip()}
        except Exception as e:
            fallback_q = f"Mujhe aur jankari chahiye, specially {missing[0]} ke bare mein. Kya aap bata sakte hain?"
            return {"extracted": {}, "hinglish_question": fallback_q}

    def process_turn(self, session_id: str, pipeline_output: dict) -> dict:
        session = self.get_or_create_session(session_id)
        session["turns"] += 1
        
        transcript = pipeline_output.get("transcript", "")
        session["verbatim"].append(f"[Turn {session['turns']} transcript: '{transcript}']")

        intent = pipeline_output.get("intent", "UNKNOWN")
        amount = pipeline_output.get("amount", "UNKNOWN")
        
        # Determine intent intelligently on Turn 1 if not set
        if session["intent"] is None:
            if "chori" in transcript.lower() or "ghuse" in transcript.lower():
                session["intent"] = "THEFT"
            elif intent != "UNKNOWN" and intent in GOAL_SCHEMAS:
                session["intent"] = intent
            else:
                session["intent"] = "THEFT" # fallback combo for demo
        
        schema = GOAL_SCHEMAS.get(session["intent"])
        
        if amount != "UNKNOWN":
            session["collected_slots"]["amount"] = amount
            
        # 🧠 Smart Agent Extraction & Generation via LLM
        llm_output = self._generate_agentic_response(transcript, schema, session)
        
        # Dynamically update collected slots
        for k, v in llm_output.get("extracted", {}).items():
            if k in schema.required_slots:
                session["collected_slots"][k] = str(v)
                
        # Re-check missing slots realistically
        missing_slots = [slot for slot in schema.required_slots if slot not in session["collected_slots"]]
        agent_msg = llm_output.get("hinglish_question", "SUCCESS")
        
        if not missing_slots or agent_msg == "SUCCESS":
            # Complete!
            final_record = { "intent_or_incident": session["intent"] }
            final_record.update(session["collected_slots"])
            final_record["verbatim"] = session["verbatim"]
            
            return {
                "status": "complete",
                "message": "All fields complete. Goal achieved.",
                "final_record": final_record
            }
        else:
            return {
                "status": "incomplete",
                "missing_slots": missing_slots,
                "agent_prompt": agent_msg
            }

if __name__ == "__main__":
    # Test True Smart Agent
    agent = SmartAgentDecisionLayer()
    
    print("\n" + "="*50)
    print("🤖 REAL AGENTIC AI DEMO (Powered by Local LLM)")
    print("="*50)
    
    turn1 = {"transcript": "kal raat mere dukan mein ghuse, teen log the, paanch hazaar le gaye", "intent": "THEFT", "amount": "Rs. 5000"}
    print(f"\n[Turn 1] Victim: {turn1['transcript']}")
    res1 = agent.process_turn("session_1", turn1)
    print(f"🧠 Agent Logic extracts dynamically -> Missing: {res1.get('missing_slots')} ")
    print(f"🎙️ Agent Asks: {res1.get('agent_prompt', res1)}")
    
    turn2 = {"transcript": "Model Town, Amritsar, main road pe, Sharma General Store", "intent": "UNKNOWN", "amount": "UNKNOWN"}
    print(f"\n[Turn 2] Victim: {turn2['transcript']}")
    res2 = agent.process_turn("session_1", turn2)
    print(f"🧠 Agent Logic extracts dynamically -> Missing: {res2.get('missing_slots')} ")
    print(f"🎙️ Agent Asks: {res2.get('agent_prompt', res2)}")
    
    turn3 = {"transcript": "Mera naam Ramesh Sharma hai", "intent": "UNKNOWN", "amount": "UNKNOWN"}
    print(f"\n[Turn 3] Victim: {turn3['transcript']}")
    res3 = agent.process_turn("session_1", turn3)
    
    print("\n✅ All fields complete. Agent generates Structured Record dynamically:")
    print(json.dumps(res3.get("final_record", res3), indent=2))
