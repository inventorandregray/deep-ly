import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import stream_tokens
from src.model_loader import load_model_and_tokenizer
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run deep'ly - a nano-vLLM clone")
    parser.add_argument("--model", type=str, required=True, help="Provide the model name or path")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt from which to generate text")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    args = parser.parse_args()

    tokenizer, model = load_model_and_tokenizer(args.model)

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(model.device)
    stream_tokens(model, tokenizer, input_ids, args.max_tokens, args.temperature, args.top_k)

if __name__ == "__main__":
    main()
