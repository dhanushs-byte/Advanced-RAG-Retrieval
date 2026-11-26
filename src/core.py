from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", "../Week7-Day1/data"))
if not DATA_DIR.is_absolute():
    DATA_DIR = (ROOT / DATA_DIR).resolve()

CACHE_DIR = ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "3"))


def tokenize(text: str) -> list[str]:
    """Simple lowercase word tokenizer for BM25."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import fitz
        with fitz.open(path) as pdf:
            return "\n".join(page.get_text("text") for page in pdf)
    if suffix == ".docx":
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""


def load_documents(folder: Path = DATA_DIR) -> list[dict[str, str]]:
    """Load supported files recursively. Evaluation JSON files are ignored."""
    if not folder.exists():
        raise FileNotFoundError(
            f"Document folder not found: {folder}\n"
            "Place your Week7-Day1 documents in ../Week7-Day1/data "
            "or set DATA_DIR in .env."
        )

    documents = []
    supported = {".pdf", ".docx", ".txt", ".md"}
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in supported:
            continue
        if path.name.lower() in {"readme.md"}:
            continue
        text = read_document(path).strip()
        if text:
            documents.append({
                "doc_id": path.stem,
                "source": str(path),
                "text": text
            })

    if not documents:
        raise RuntimeError(
            f"No PDF, DOCX, TXT or MD documents found in {folder}. "
            "Copy your Week7-Day1 source documents into that folder."
        )
    return documents


def chunk_documents(
    documents: list[dict[str, str]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[dict[str, Any]]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")
    chunks = []
    for doc in documents:
        text = re.sub(r"\s+", " ", doc["text"]).strip()
        start = 0
        part = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            content = text[start:end].strip()
            if content:
                chunks.append({
                    "chunk_id": f'{doc["doc_id"]}__{part:03d}',
                    "doc_id": doc["doc_id"],
                    "source": doc["source"],
                    "text": content
                })
                part += 1
            if end >= len(text):
                break
            start = end - overlap
    return chunks


def build_embeddings(chunks: list[dict[str, Any]]) -> tuple[SentenceTransformer, np.ndarray]:
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    vectors = model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=True
    )
    return model, np.asarray(vectors, dtype=np.float32)


def vector_search(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: list[dict[str, Any]],
    top_k: int = TOP_K
) -> list[str]:
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = embeddings @ q
    order = np.argsort(scores)[::-1][:top_k]
    return [chunks[int(i)]["chunk_id"] for i in order]


def build_bm25(chunks: list[dict[str, Any]]) -> BM25Okapi:
    tokenized = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized)


def bm25_search(
    query: str, bm25: BM25Okapi, chunks: list[dict[str, Any]],
    top_k: int = TOP_K
) -> list[str]:
    scores = bm25.get_scores(tokenize(query))
    order = np.argsort(scores)[::-1][:top_k]
    return [chunks[int(i)]["chunk_id"] for i in order]


def rrf(
    rankings: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion.
    rank is 1-based: score += weight / (k + rank).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("One weight is required for each ranking list.")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + float(weight) / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def get_chunk_map(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {c["chunk_id"]: c for c in chunks}


def is_hit(
    result_ids: list[str],
    chunk_map: dict[str, dict[str, Any]],
    expected_terms: list[str]
) -> bool:
    """Hit if any retrieved top-k chunk contains any expected phrase."""
    terms = [t.lower() for t in expected_terms]
    for chunk_id in result_ids:
        text = chunk_map[chunk_id]["text"].lower()
        if any(term in text for term in terms):
            return True
    return False


def hit_rate(hit_values: list[bool]) -> float:
    return sum(hit_values) / len(hit_values) if hit_values else 0.0


class LLMHelper:
    """Optional OpenAI-compatible API client, with a deterministic local fallback."""

    def __init__(self):
        self.enabled = os.getenv("USE_LLM", "false").lower() == "true"
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model_name = os.getenv("LLM_MODEL", "liquid/lfm-3-1.7b:free")
        self.input_rate = float(os.getenv("INPUT_COST_PER_1M", "0"))
        self.output_rate = float(os.getenv("OUTPUT_COST_PER_1M", "0"))
        self.input_tokens = 0
        self.output_tokens = 0
        self.estimated_cost = 0.0
        self.api_latency = 0.0
        self.client = None

        if self.enabled:
            if not self.api_key:
                raise ValueError("USE_LLM=true but OPENAI_API_KEY is empty in .env")
            from openai import OpenAI
            base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        self.api_latency += time.perf_counter() - start
        usage = getattr(response, "usage", None)
        if usage:
            self.input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        self.estimated_cost = (
            self.input_tokens * self.input_rate
            + self.output_tokens * self.output_rate
        ) / 1_000_000
        return response.choices[0].message.content or ""

    def rewrite(self, question: str) -> list[str]:
        if not self.enabled:
            return local_rewrites(question)
        prompt = (
            "Rewrite the user's question into exactly 3 alternative search queries. "
            "Expand abbreviations (for example, WFH means work from home). "
            "Preserve the original intent. Return ONLY a JSON array of 3 strings."
        )
        raw = self._chat(prompt, question)
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                values = [str(x).strip() for x in data if str(x).strip()]
                if len(values) >= 3:
                    return values[:3]
        except json.JSONDecodeError:
            pass
        # Recover three lines if provider does not return valid JSON.
        lines = [re.sub(r"^[\d\-\.\)\s]+", "", x).strip()
                 for x in raw.splitlines()]
        lines = [x for x in lines if x]
        return (lines + local_rewrites(question))[:3]

    def hyde(self, question: str) -> str:
        if not self.enabled:
            return local_hypothetical_answer(question)
        prompt = (
            "Write a short hypothetical answer (2-4 sentences) to the question. "
            "You do not have access to company documents. Do not claim that the "
            "answer is verified. Include likely terminology that a relevant "
            "policy document might contain."
        )
        return self._chat(prompt, question).strip()


def local_rewrites(question: str) -> list[str]:
    q = question.strip()
    expanded = re.sub(r"\bWFH\b", "work from home", q, flags=re.I)
    expanded = re.sub(r"\bPTO\b", "paid time off", expanded, flags=re.I)
    expanded = re.sub(r"\bHR\b", "human resources", expanded, flags=re.I)
    return [
        expanded,
        f"What does the company policy say about {expanded}?",
        f"Rules, eligibility, process and requirements for: {expanded}"
    ]


def local_hypothetical_answer(question: str) -> str:
    q = question.lower()
    if "wfh" in q or "remote" in q or "work from home" in q:
        return ("A work from home policy usually describes remote-work eligibility, "
                "manager approval, permitted days, attendance and availability.")
    if "notice" in q or "resignation" in q or "leaving" in q:
        return ("An employee resignation policy usually specifies the required "
                "notice period, how to submit resignation and manager or HR procedures.")
    if "sick" in q or "ill" in q or "unwell" in q:
        return ("A sick leave policy explains medical leave entitlement, notification "
                "to the manager and any supporting medical documentation.")
    if "casual" in q or "annual" in q or "leave" in q or "time off" in q:
        return ("An employee leave policy explains leave categories, annual allowance, "
                "eligibility, application and approval by the reporting manager.")
    return (f"A company policy relevant to '{question}' would describe the applicable "
            "rules, employee eligibility, process, approvals and responsibilities.")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
