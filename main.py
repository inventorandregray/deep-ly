import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import stream_tokens
from src.model_loader import load_model_and_tokenizer
import argparse

def main():
    parser = argparse.ArgumentParser(description="deep'ly চালান - একটি nano-vLLM ক্লোন")
    parser.add_argument("--model", type=str, required=True, help="মডেলের নাম অথবা পাথ দিন")
    parser.add_argument("--prompt", type=str, required=True, help="যে প্রম্পট থেকে জেনারেট করবেন")
    parser.add_argument("--max_tokens", type=int, default=100, help="সর্বোচ্চ কতটি নতুন টোকেন জেনারেট হবে")
    parser.add_argument("--temperature", type=float, default=1.0, help="স্যাম্পলিং টেম্পারেচার")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k স্যাম্পলিং")
    args = parser.parse_args()

    tokenizer, model = load_model_and_tokenizer(args.model)

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(model.device)
    stream_tokens(model, tokenizer, input_ids, args.max_tokens, args.temperature, args.top_k)

if __name__ == "__main__":
    main()
