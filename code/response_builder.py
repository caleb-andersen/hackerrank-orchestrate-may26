from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List, Sequence

from retriever import RetrievedDocument, query_tokens, tokenize
from router import LOW_CONFIDENCE_REASON, RouteDecision


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
ACTION_TERMS = ("contact", "call", "log in", "go to", "visit", "report", "click", "select", "navigate")
ACTION_RE = re.compile(r"\b(contact|call|log in|go to|visit|report|click|select|navigate)\b", re.IGNORECASE)
UNRELATED_SENTENCE_MARKERS = (
    "session name",
    "your session",
    "terminal",
    "command line",
    "keyboard shortcut",
    "press enter",
    "claude code",
    "power user",
    "launch guide",
)
CARD_RISK_ESCALATION_REASON = "fraud, identity, or card-risk issues require human support"


@dataclass(frozen=True)
class BuiltResponse:
    response: str
    justification: str


def _clean_sentence(sentence: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", sentence)
    cf_email_link = r"\[(?:\\.|[^\]])+\]\(/cdn-cgi/l/email-protection[^)]*\)"
    cleaned = re.sub(rf"\s*(?:please\s+)?contact us at\s+{cf_email_link}", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(cf_email_link, " ", cleaned)
    cleaned = re.sub(r"/cdn-cgi/l/email-protection\S*", " ", cleaned)
    cleaned = re.sub(r"\[\]\([^)]+\)", " ", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = cleaned.replace("**", "")
    cleaned = re.sub(r",\s*([.!?])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -\t\r\n")
    if cleaned.startswith("#") or cleaned.startswith("---"):
        return ""
    if cleaned.count("|") >= 2:
        return ""
    if cleaned.lower().startswith(("review these resources", "related articles", "subscribe to updates")):
        return ""
    if cleaned.lower().startswith(("title:", "source_url:", "final_url:", "last_updated", "description:")):
        return ""
    return cleaned


def _extract_sentences(text: str) -> List[str]:
    sentences = []
    for raw in SENTENCE_SPLIT_RE.split(text):
        sentence = _clean_sentence(raw)
        if len(sentence) < 20:
            continue
        sentences.append(sentence)
    return sentences


def _mentions_crawling_topic(text: str) -> bool:
    lower = text.lower()
    return any(term in lower for term in ("crawl", "crawling", "crawler", "robots.txt", "claudebot", "opt out", "opt-out", "disallow"))


def _mentions_data_use_topic(text: str) -> bool:
    lower = text.lower()
    has_data = any(term in lower for term in ("data", "privacy", "retention"))
    has_training = any(term in lower for term in ("train", "training", "model", "models", "improve"))
    return has_data and has_training or "how long will the data be used" in lower


def _topic_bonus(query: str, sentence: str) -> int:
    query_lower = query.lower()
    sentence_lower = sentence.lower()
    bonus = 0

    if _mentions_crawling_topic(query_lower) and any(
        term in sentence_lower for term in ("crawl", "crawler", "robots.txt", "claudebot", "disallow", "opt out")
    ):
        bonus += 3

    if _mentions_data_use_topic(query_lower) and any(
        term in sentence_lower
        for term in ("data retention", "retention period", "train our models", "used to train", "default", "retained indefinitely")
    ):
        bonus += 3

    if "pause" in query_lower and "subscription" in query_lower:
        if "pause" in sentence_lower or "cancel plan" in sentence_lower:
            bonus += 4
        if "cancel subscription" in sentence_lower or "cancel your subscription" in sentence_lower:
            bonus -= 3

    return bonus


def _rank_sentences(query: str, docs: Sequence[RetrievedDocument]) -> List[tuple[int, str]]:
    query_token_set = set(query_tokens(query))
    ranked = []
    seen = set()

    for result in docs:
        for sentence in _extract_sentences(result.document.content):
            if sentence.lower() == result.title.lower():
                continue
            sentence_tokens = set(tokenize(sentence))
            overlap = len(query_token_set & sentence_tokens)
            if overlap < 2:
                continue
            bonus = 0
            lower_sentence = sentence.lower()
            if ACTION_RE.search(lower_sentence):
                bonus += 1
            bonus += _topic_bonus(query, sentence)
            score = overlap + bonus
            normalized = lower_sentence
            if normalized in seen:
                continue
            seen.add(normalized)
            ranked.append((score, sentence))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked


def _top_sentences(query: str, docs: Sequence[RetrievedDocument], limit: int = 2) -> List[str]:
    return [sentence for _, sentence in _rank_sentences(query, docs)[:limit]]


def _safe_next_step(query: str, docs: Sequence[RetrievedDocument]) -> str:
    query_token_set = set(query_tokens(query))
    for score, step in _rank_sentences(query, docs):
        lower = step.lower()
        overlap = len(query_token_set & set(tokenize(step)))
        if score < 3 or overlap < 2:
            continue
        if any(marker in lower for marker in UNRELATED_SENTENCE_MARKERS):
            continue
        if ACTION_RE.search(lower):
            return step
    return ""


def _fallback_invalid_response(query: str) -> str:
    lower = query.lower()
    if "delete all files" in lower or "delete" in lower and "system" in lower:
        return "I can’t help with deleting files or damaging a system."
    if any(
        term in lower
        for term in (
            "internal rules",
            "hidden logic",
            "chain of thought",
            "documents retrieved",
            "retrieved documents",
            "règles internes",
            "documents récupérés",
            "logique exacte",
            "reglas internas",
            "documentos internos",
            "lógica exacta",
        )
    ):
        response = "I can’t share internal rules, retrieved documents, or the agent’s decision logic."
        if any(term in lower for term in ("visa", "card", "carte", "tarjeta", "issuer", "bloquée", "bloqueada", "blocked")):
            response += " If you have a real card issue, please contact your issuer using the number on the back of your card."
        return response
    if "thank you" in lower:
        return "You’re welcome."
    return "This request is outside the scope of the support agent."


def build_response(
    issue: str,
    subject: str,
    company: str,
    route: RouteDecision,
    retrieved_docs: Sequence[RetrievedDocument],
) -> BuiltResponse:
    query = f"{subject} {issue}".strip()

    if route.request_type == "invalid":
        response = _fallback_invalid_response(query)
        justification = "Handled as an invalid or out-of-scope request without escalation because no account or payment action is required."
        return BuiltResponse(response=response, justification=justification)

    if route.status == "escalated":
        next_step = ""
        if route.escalation_reason == CARD_RISK_ESCALATION_REASON:
            next_step = _safe_next_step(query, retrieved_docs)
        response = f"This request needs human support because {route.escalation_reason}."
        if next_step:
            response += f" A safe next step from the documentation is: {next_step}"
        justification = f"Escalated because {route.escalation_reason}."
        return BuiltResponse(response=response, justification=justification)

    evidence = _top_sentences(query, retrieved_docs, limit=2)
    if evidence:
        response = " ".join(evidence)
    else:
        response = "I could not find enough grounded information in the provided support corpus to answer this safely."

    if retrieved_docs:
        source_bits = ", ".join(doc.product_area or doc.title for doc in retrieved_docs[:2])
        justification = f"Replied using the top retrieved documentation from {source_bits}."
    else:
        justification = "Replied conservatively because the request was low-risk."

    return BuiltResponse(response=response, justification=justification)
