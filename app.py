import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI(title="Gemma 3 270M API")

# Model identifier
MODEL_ID = "google/gemma-3-270m-it"

# Initialize model and tokenizer globally for caching
print("Loading model into memory...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,  # Reduces RAM footprint
    device_map="cpu"            # Forces CPU execution for Render
)
print("Model loaded successfully!")

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 128

@app.get("/")
def health_check():
    return {"status": "healthy", "model": MODEL_ID}

@app.post("/generate")
def generate_text(payload: ChatRequest):
    # Apply standard chat template for Gemma instruction-tuned variants
    messages = [{"role": "user", "content": payload.prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to("cpu")
    
    outputs = model.generate(
        input_ids, 
        max_new_tokens=payload.max_tokens,
        do_sample=True,
        temperature=0.7
    )
    
    # Decode and return response excluding the original prompt
    generated_ids = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return {"response": response.strip()}
