# GodeusAI - Conversations with the infinite, locally hosted.

GodeusAI is a spiritual assistant chatbot powered by a fine-tuned Mistral-7B language model with LoRA adapters. It provides answers to spiritual and philosophical questions, and can be run locally for private, offline use.

---

## Features

- Chatbot interface for spiritual Q&A
- Fine-tuned on custom spiritual/philosophical data
- LoRA adapter for efficient model adaptation
- Utilities for context-based question answering
- Example notebook for data preparation and training

---

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Chatbot

```bash
python inference.py
```

You’ll see a prompt:  
`💉 Welcome to GodeusAI — your spiritual assistant. Type 'exit' to quit.`

---

## Project Structure

| File/Folder              | Purpose                                                      |
|--------------------------|--------------------------------------------------------------|
| `inference.py`           | Main script to chat with the model                           |
| `rag_utils.py`           | Utilities for loading model and context-based QA             |
| `few_shots_qa.jsonl`     | Example Q&A pairs for few-shot prompting                     |
| `adapter_model.safetensors` | LoRA adapter weights (required for inference)             |
| `adapter_config.json`    | Configuration for the LoRA adapter                           |
| `GodeusAI.ipynb`         | Notebook for data prep, training, and experimentation        |
| `requirements.txt`       | Python dependencies                                          |

---

## Data & Training

- **few_shots_qa.jsonl**: Example format for few-shot Q&A pairs.
- **GodeusAI.ipynb**:  
  - Prepares data from PDFs  
  - Chunks and formats into JSONL  
  - Trains LoRA adapters on Mistral-7B

---

## Model

- **Base model**: `mistralai/Mistral-7B-v0.1`
- **Adapter**: LoRA (config in `adapter_config.json`, weights in `adapter_model.safetensors`)

---

## License

This project is for research and educational purposes.  
Refer to the base model and dataset licenses for usage restrictions. 