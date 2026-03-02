# Semantic Memory Kit

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org)
[![Model](https://img.shields.io/badge/model-all--MiniLM--L6--v2-green)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

Lightweight semantic search over your AI agent's memory files. No vector database. No API calls. Runs on CPU in ~85ms per query.

> **Packaged version available:** Pre-bundled zip with extended examples and an agent integration demo — [download on Gumroad](https://sagnelli.gumroad.com/l/rplxp) (€29).

## The Problem

Most AI agents treat memory as a file you append to and eventually load into context. This fails in two ways:

1. **Too much context** — loading everything hits token limits and costs money
2. **Keyword search misses intent** — searching for "payment setup" won't find "configured Stripe for billing"

## The Solution

Encode your memory files as local embeddings. Search by meaning, not keywords. Return only the relevant chunks.

```python
from semantic_memory import SemanticMemory

mem = SemanticMemory("~/.agent/memory")
mem.index()

results = mem.query("Stripe payment integration", top_k=3)
# → [0.847] stripe_notes.md: Set up Stripe webhook handler. Use idempotency keys...
# → [0.731] payments.md: Stripe requires HTTPS in live mode...
# → [0.612] decisions.md: Chose Stripe over PayPal due to better API documentation...
```

## Install

```bash
pip install sentence-transformers numpy
```

Then copy `semantic_memory.py` into your project.

## Usage

### Index your memory files

```python
from semantic_memory import SemanticMemory

mem = SemanticMemory("~/.agent/memory")
mem.index()
# → Indexing 205 chunks from 12 files...
# → Index built. 205 chunks ready.
```

Supports `.md`, `.txt`, `.json` files. Index is cached to `.semantic_index.json` — no re-indexing unless files change.

### Query by meaning

```python
results = mem.query("what did we decide about the database?", top_k=5)
for r in results:
    print(f"[{r.score:.3f}] {r.source}: {r.text[:100]}")
```

### Get formatted context for prompt injection

```python
context = mem.query_and_format("current project priorities", top_k=4)

# Inject directly into your model's system prompt:
response = client.messages.create(
    model="claude-haiku-4-5",
    system=f"You are an assistant.\n\nMemory:\n{context}",
    messages=[{"role": "user", "content": user_message}]
)
```

### Full example with Ollama

```python
import ollama
from semantic_memory import SemanticMemory

mem = SemanticMemory("~/.agent/memory")
mem.index()

def chat(message):
    context = mem.query_and_format(message, top_k=4)
    return ollama.chat(
        model="mistral",
        messages=[
            {"role": "system", "content": f"Memory:\n{context}"},
            {"role": "user", "content": message}
        ]
    )["message"]["content"]
```

### CLI

```bash
# Build index
python semantic_memory.py ~/.agent/memory --index

# Query
python semantic_memory.py ~/.agent/memory "what is the status of the API project?"
```

## How It Works

```
Memory files (.md / .txt / .json)
        │
        ▼
  ┌───────────┐
  │  Chunker  │  Split into ~400-char overlapping segments
  └─────┬─────┘
        │
        ▼
  ┌───────────┐
  │  Encoder  │  all-MiniLM-L6-v2 (22MB, CPU, ~85ms/query)
  └─────┬─────┘
        │
        ▼
  ┌──────────────────────┐
  │  .semantic_index.json│  Chunks + embeddings stored locally
  └─────┬────────────────┘
        │
  Query │  encode → cosine similarity → top-K
        ▼
  Ranked relevant chunks (with score + source file)
        │
        ▼
  Inject into model context
```

## Performance

| Operation | Time (M1 MacBook Air, 205 chunks) |
|-----------|----------------------------------|
| First index build | ~3.2 seconds |
| Index load from cache | ~0.4 seconds |
| Query (encode + similarity) | ~85ms |
| Index file size | ~1.4MB |
| Model download (one-time) | 22MB |
| RAM while loaded | ~180MB |

## When to Use This vs a Vector Database

**Use this when:**
- Fewer than ~10,000 memory chunks
- Zero infrastructure — no server, no Docker, no account
- Privacy matters (all data stays local)
- Single-process agent

**Use Chroma / Pinecone / pgvector when:**
- More than 10,000 chunks
- Multiple processes need concurrent access
- Sub-10ms latency required at scale

## Requirements

```
sentence-transformers>=2.2.0
numpy>=1.21.0
```

Python 3.8+. No other dependencies.

## License

MIT

---

Extended examples including multi-agent setups, custom indexing patterns, and OpenAI function calling integration: [Gumroad](https://sagnelli.gumroad.com/l/rplxp)
