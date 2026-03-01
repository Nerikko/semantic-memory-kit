"""
Semantic Memory Kit for AI Agents
----------------------------------
Lightweight semantic search over your agent's memory files.
No vector database required. Uses local embeddings + numpy cosine similarity.

Usage:
    mem = SemanticMemory("~/.agent/memory")
    mem.index()                              # build/update index
    results = mem.query("Stripe integration", top_k=3)
    for r in results: print(r.score, r.text)
"""

import os, json, hashlib, glob
from pathlib import Path
from dataclasses import dataclass
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"   # 22MB, runs on CPU, fast
CHUNK_SIZE  = 400                   # chars per chunk
CHUNK_OVERLAP = 80                  # overlap between chunks

@dataclass
class MemoryChunk:
    text: str
    source: str        # file path
    line_start: int
    score: float = 0.0

class SemanticMemory:
    def __init__(self, memory_dir: str, index_path: str = None):
        self.memory_dir = Path(memory_dir).expanduser()
        self.index_path = Path(index_path or (self.memory_dir / ".semantic_index.json"))
        self._model = None
        self._chunks: List[MemoryChunk] = []
        self._embeddings: np.ndarray = None
        self._load_index()

    @property
    def model(self):
        if self._model is None:
            print("Loading embedding model (first run only)...")
            self._model = SentenceTransformer(MODEL_NAME)
        return self._model

    def _chunk_text(self, text: str, source: str) -> List[MemoryChunk]:
        """Split text into overlapping chunks."""
        chunks = []
        lines = text.split('\n')
        current, line_start, length = [], 0, 0
        for i, line in enumerate(lines):
            current.append(line)
            length += len(line) + 1
            if length >= CHUNK_SIZE:
                chunk_text = '\n'.join(current).strip()
                if chunk_text:
                    chunks.append(MemoryChunk(chunk_text, source, line_start))
                # Keep overlap
                overlap_lines = current[-3:] if len(current) > 3 else current
                line_start = i - len(overlap_lines) + 1
                current = overlap_lines
                length = sum(len(l) + 1 for l in overlap_lines)
        if current:
            chunk_text = '\n'.join(current).strip()
            if chunk_text:
                chunks.append(MemoryChunk(chunk_text, source, line_start))
        return chunks

    def index(self, patterns: List[str] = None):
        """Scan memory files and build/update the semantic index."""
        if patterns is None:
            patterns = ["**/*.md", "**/*.txt", "**/*.json"]
        
        all_chunks = []
        checksums = {}

        for pattern in patterns:
            for path in self.memory_dir.glob(pattern):
                if path.name.startswith('.'):
                    continue
                try:
                    text = path.read_text(encoding='utf-8', errors='ignore')
                    checksum = hashlib.md5(text.encode()).hexdigest()
                    checksums[str(path)] = checksum
                    chunks = self._chunk_text(text, str(path))
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Skip {path}: {e}")

        if not all_chunks:
            print("No files found to index.")
            return

        print(f"Indexing {len(all_chunks)} chunks from {len(checksums)} files...")
        texts = [c.text for c in all_chunks]
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        
        self._chunks = all_chunks
        self._embeddings = embeddings
        self._save_index(checksums)
        print(f"Index built. {len(all_chunks)} chunks ready.")

    def query(self, query: str, top_k: int = 5, min_score: float = 0.3) -> List[MemoryChunk]:
        """Find the most semantically relevant memory chunks for a query."""
        if self._embeddings is None or len(self._chunks) == 0:
            print("No index. Run .index() first.")
            return []
        
        q_embedding = self.model.encode([query])[0]
        # Cosine similarity
        norms = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(q_embedding)
        norms = np.where(norms == 0, 1e-9, norms)
        scores = np.dot(self._embeddings, q_embedding) / norms
        
        top_indices = np.argsort(scores)[::-1][:top_k * 2]
        results = []
        seen_sources = set()
        for i in top_indices:
            score = float(scores[i])
            if score < min_score:
                break
            chunk = self._chunks[i]
            # Deduplicate by source if same file appears many times
            key = f"{chunk.source}:{chunk.line_start}"
            if key not in seen_sources:
                results.append(MemoryChunk(
                    text=chunk.text,
                    source=chunk.source,
                    line_start=chunk.line_start,
                    score=round(score, 3)
                ))
                seen_sources.add(key)
            if len(results) >= top_k:
                break
        return results

    def query_and_format(self, query: str, top_k: int = 5) -> str:
        """Query and return formatted string for injection into model context."""
        results = self.query(query, top_k=top_k)
        if not results:
            return f"[SemanticMemory: no relevant context found for '{query}']"
        
        parts = [f"[SemanticMemory: {len(results)} relevant chunks for '{query}']\n"]
        for i, r in enumerate(results, 1):
            source = Path(r.source).name
            parts.append(f"--- [{i}] {source} (score: {r.score}) ---")
            parts.append(r.text)
            parts.append("")
        return '\n'.join(parts)

    def _save_index(self, checksums: dict):
        data = {
            "checksums": checksums,
            "chunks": [{"text": c.text, "source": c.source, "line_start": c.line_start} 
                      for c in self._chunks],
            "embeddings": self._embeddings.tolist()
        }
        self.index_path.write_text(json.dumps(data))

    def _load_index(self):
        if not self.index_path.exists():
            return
        try:
            data = json.loads(self.index_path.read_text())
            self._chunks = [MemoryChunk(**c) for c in data["chunks"]]
            self._embeddings = np.array(data["embeddings"])
        except Exception as e:
            print(f"Could not load index: {e}")


# CLI
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: semantic_memory.py <memory_dir> <query>")
        print("       semantic_memory.py <memory_dir> --index")
        sys.exit(1)
    
    memory_dir = sys.argv[1]
    mem = SemanticMemory(memory_dir)
    
    if sys.argv[2] == "--index":
        mem.index()
    else:
        query = ' '.join(sys.argv[2:])
        print(mem.query_and_format(query, top_k=5))


# Auto-index for Datis workspace
def index_datis_workspace():
    """Index the full Datis workspace for semantic search."""
    mem = SemanticMemory("/Users/enrico/.openclaw/workspace")
    mem.index(patterns=["*.md", "memory/*.md", "knowledge/**/*.md", "research/*.md"])
    return mem

