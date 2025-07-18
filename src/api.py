from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.model_loader import load_model_and_tokenizer
from src.utils import stream_tokens
import torch
import asyncio

app = FastAPI(title="deep'ly LLM Inference API")

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 50

# Load model/tokenizer cache to avoid reloading on every request
_loaded_models = {}

def get_model_tokenizer(model_name):
    if model_name not in _loaded_models:
        tokenizer, model = load_model_and_tokenizer(model_name)
        _loaded_models[model_name] = (tokenizer, model)
    return _loaded_models[model_name]

@app.post("/generate")
async def generate(req: GenerateRequest):
    tokenizer, model = get_model_tokenizer(req.model)

    input_ids = tokenizer(req.prompt, return_tensors="pt").input_ids.to(model.device)

    # Collect tokens as a list to return as response
    generated_tokens = []

    # We wrap stream_tokens to collect output tokens instead of printing
    device = input_ids.device
    past_key_values = None
    generated = input_ids

    with torch.no_grad():
        for _ in range(req.max_tokens):
            outputs = model(
                input_ids=generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = outputs.logits[:, -1, :] / req.temperature

            if req.top_k > 0:
                top_k = min(req.top_k, logits.size(-1))
                values, _ = torch.topk(logits, top_k)
                min_values = values[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_values,
                    torch.full_like(logits, fill_value=-float("Inf")),
                    logits
                )

            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

            decoded_token = tokenizer.decode(next_token[0], skip_special_tokens=True)
            generated_tokens.append(decoded_token)

            past_key_values = outputs.past_key_values

    return {"generated_text": "".join(generated_tokens)}

