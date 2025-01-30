# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json

import litellm  # Minimalistic LLM library (you can replace with huggingface, etc.)

app = FastAPI()

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 512

@app.post("/v1/inference")
async def inference(request: InferenceRequest):
    """
    Inference endpoint using deepseek-r1-distill-7b (uncensored).
    """
    response = litellm.completion(
        model="deepseek/deepseek-r1-distill-7b",
        messages=[{"role": "user", "content": request.prompt}],
        max_tokens=request.max_tokens
    )
    return JSONResponse(content=response)

@app.post("/v1/finetune")
async def finetune(file: UploadFile = File(...)):
    """
    Fine-tune endpoint that accepts a JSONL file.
    """
    if not file.filename.endswith('.jsonl'):
        return JSONResponse(
            status_code=400,
            content={"error": "Only .jsonl files are accepted for fine-tuning"}
        )

    # Read lines from uploaded file
    data = [json.loads(line) for line in file.file]

    # Perform or schedule a fine-tuning job here (simplified placeholder)
    # You can integrate with your training pipeline below.
    
    return JSONResponse(content={"status": "Fine-tuning request received", "samples": len(data)})