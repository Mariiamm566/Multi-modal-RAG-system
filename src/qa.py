from __future__ import annotations
from typing import Any
import requests


def _citation(item: dict[str, Any]) -> str:
    channel = item.get("retrieval_channel")
    suffix = f", via {channel}" if channel else ""
    return f"{item['file_name']} p.{item['page']}{suffix}"


def _format_evidence_line(item: dict[str, Any]) -> str:
    text = item["text"].replace("\n", " ").strip()
    text = text[:420]
    modality = item["modality"].upper()
    return f"[{modality}] {text} [{_citation(item)}]"


def _collect_citations(contexts: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    citations: list[str] = []
    for item in contexts:
        c = _citation(item)
        if c not in seen:
            seen.add(c)
            citations.append(c)
    return citations


def _best_contexts(contexts: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    return sorted(contexts, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:limit]


def _extractive_answer(query: str, contexts: list[dict[str, Any]]) -> str:
    if not contexts:
        return "I could not find grounded evidence in the indexed documents to answer this question."

    top_items = _best_contexts(contexts, limit=3)
    summary_parts: list[str] = []

    for item in top_items:
        text = item["text"].replace("\n", " ").strip()
        if len(text) > 260:
            text = text[:257] + "..."
        if item["modality"] == "table":
            summary_parts.append(f"Table evidence indicates: {text}")
        elif item["modality"] == "image":
            summary_parts.append(f"Image-derived evidence suggests: {text}")
        else:
            summary_parts.append(text)

    citations = _collect_citations(top_items)
    answer_body = " ".join(summary_parts).strip()
    citation_lines = "\n".join(f"- {c}" for c in citations)

    return f"{answer_body}\n\nSources:\n{citation_lines}"


def _api_answer(query: str, contexts: list[dict[str, Any]], endpoint: str, api_key: str, model: str) -> str:
    evidence = [_format_evidence_line(item) for item in contexts]
    prompt = (
        "Answer the user's question using only the evidence below. "
        "Write one concise grounded paragraph. "
        "Then add a section titled Sources with bullet citations copied from the evidence. "
        "If evidence is insufficient, say so clearly.\n\n"
        f"Question: {query}\n\nEvidence:\n" + "\n".join(evidence)
    )
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {"model": model, "input": prompt}
    response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        for key in ["output_text", "answer", "text"]:
            if key in data and isinstance(data[key], str) and data[key].strip():
                return data[key].strip()
    return _extractive_answer(query, contexts)


def answer_question(query: str, contexts: list[dict[str, Any]], llm_mode: str, endpoint: str, api_key: str, model: str) -> str:
    if llm_mode == "api" and endpoint:
        try:
            return _api_answer(query, contexts, endpoint, api_key, model)
        except Exception:
            return _extractive_answer(query, contexts)
    return _extractive_answer(query, contexts)