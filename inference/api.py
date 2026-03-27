import uvicorn
from fastapi import FastAPI, UploadFile, File
import tempfile
import os

from inference.pipeline import EntropicPipeline

app = FastAPI(title="Entropic ASR Engine")
pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    # Load all heavy ML models into VRAM once upon server start
    pipeline = EntropicPipeline()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Save the uploaded byte stream to a temporary .wav file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Run the blazing fast pipeline (ASR -> ITN -> Intent)
        result = pipeline.transcribe(tmp_path)
    finally:
        os.unlink(tmp_path)
        
    return result

if __name__ == "__main__":
    uvicorn.run("inference.api:app", host="0.0.0.0", port=8000, reload=False)
