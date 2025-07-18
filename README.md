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
