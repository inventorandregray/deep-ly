import torch
import torch.nn.functional as F

def stream_tokens(model, tokenizer, input_ids, max_tokens, temperature=1.0, top_k=50):
    device = input_ids.device
    past_key_values = None
    generated = input_ids

    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(
                input_ids=generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = outputs.logits[:, -1, :] / temperature

            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, top_k)
                min_values = values[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_values,
                    torch.full_like(logits, fill_value=-float("Inf")),
                    logits
                )

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

            decoded_token = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(decoded_token, end="", flush=True)

            past_key_values = outputs.past_key_values
