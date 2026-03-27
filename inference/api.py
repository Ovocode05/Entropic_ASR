import uvicorn
import tempfile
import os
import time
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from inference.pipeline import EntropicPipeline
from inference.agent import SmartAgentDecisionLayer

app = FastAPI(title="Entropic Agentic ASR")
asr_pipeline = None
llm_agent = None

@app.on_event("startup")
async def startup_event():
    global asr_pipeline, llm_agent
    print("\n🚀 [SYSTEM BOOT] Loading Full Agentic Pipeline...")
    asr_pipeline = EntropicPipeline()
    llm_agent = SmartAgentDecisionLayer()
    print("✅ [SYSTEM READY] FastAPI Server starting on 0.0.0.0:8000...\n")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Legacy Single-Turn Transcribe
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        result = asr_pipeline.transcribe(tmp_path)
    finally:
        os.unlink(tmp_path)
    return result

@app.post("/chat")
async def chat_endpoint(
    session_id: str = Form(...),
    audio: UploadFile = File(...)
):
    """
    Stateful conversational endpoint.
    Takes a session_id and a user voice file -> returns the LLM's Hinglish response.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name
        
    try:
        t0 = time.time()
        
        # 1. Run ASR Pipeline (Whisper -> ITN -> Intent)
        transcription_result = asr_pipeline.transcribe(tmp_path)
        
        # Clean the intent string for the Agent (remove ' (Rejected: low confidence)')
        raw_intent = transcription_result["intent"].split(" ")[0]
        transcription_result["intent"] = raw_intent
        
        # 2. Agent Decision Layer (Local LLM prompt generation + Stateful Extraction)
        agent_response = llm_agent.process_turn(session_id, transcription_result)
        
        latency = round((time.time() - t0) * 1000, 2)
        
        # 3. Construct Unified Multi-Turn Response
        return JSONResponse({
            "session_id": session_id,
            "pipeline_state": transcription_result,
            "agent_state": agent_response,
            "total_latency_ms": latency
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    uvicorn.run("inference.api:app", host="0.0.0.0", port=8000, reload=False)
