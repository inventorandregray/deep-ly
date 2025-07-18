# deep'ly — Minimal LLM Inference Engine

**deep'ly** is a minimal, fast, and transparent inference engine for running large language models (LLMs) with streaming output. Inspired by [vLLM](https://github.com/vllm-project/vllm), but built from scratch with under 150 lines of Python, deep'ly focuses on simplicity, performance, and open accessibility.

![License](https://img.shields.io/github/license/yourusername/deeply)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Models](https://img.shields.io/badge/models-HuggingFace-green)

---

## 🚀 Features

- 🔁 **Streaming token-by-token output**
- ⚡ **Fast GPU or CPU inference**
- 🔍 **Readable, modifiable codebase (~150 LOC)**
- 🔢 **Supports temperature and top-k sampling**
- 🧩 **Compatible with any Hugging Face `AutoModelForCausalLM`**

---

## 📦 Installation

Make sure you have Python 3.8+ installed.

```bash
git clone https://github.com/yourusername/deeply.git
cd deeply
pip install torch transformers

---

🧠 Usage
Run a model from Hugging Face with your custom prompt:

bash
Copy
Edit
python main.py \
  --model "tiiuae/falcon-7b-instruct" \
  --prompt "Once upon a time in Belize," \
  --max_tokens 100 \
  --temperature 0.8 \
  --top_k 50
You can use any of the following Hugging Face models (and more):

gpt2

mistralai/Mistral-7B-Instruct-v0.1

meta-llama/Llama-2-7b-chat-hf

google/gemma-7b

NousResearch/Nous-Hermes-2-Mistral-7B

🧱 Project Structure
deeply/
├── main.py               # Entry point with CLI
├── src/
│   ├── model_loader.py   # Loads model/tokenizer
│   └── utils.py          # Core inference loop
└── LICENSE               # MIT License
🛠️ Development Goals
Add web server (FastAPI)

Add batching / KV caching profiler

Enable quantized model support (e.g., GGUF or AWQ)

Docker container

📄 License
MIT License © 2025 Andre Gray

🌐 Acknowledgments
Nano _vLLM for inspiration

Hugging Face Transformers for the model zoo


### ✅ Next Steps:

1. Replace `"yourusername"` with your actual GitHub username.
2. Create a `.gitignore` (I can help generate it).
3. Optionally add a `requirements.txt` file for easy install:
torch
Transformers
Vbnet

---

## 📌 How to build and run with Docker
bash

docker build -t deeply-llm .
docker run -p 8000:8000 deeply-llm
Then open http://localhost:8000/docs to test the interactive Swagger UI.



Summary:
●	Dockerfile uses Python 3.10 slim, installs dependencies, exposes port 8000, and runs FastAPI.

●	FastAPI app exposes /generate POST endpoint to run text generation.

●	Models are cached in memory to improve performance.

●	Returns generated text as JSON.
