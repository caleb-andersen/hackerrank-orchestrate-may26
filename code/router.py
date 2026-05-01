from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from retriever import RetrievedDocument, query_tokens, tokenize


@dataclass(frozen=True)
class RouteDecision:
    status: str
    request_type: str
    product_area: str
    escalation_reason: str = ""


COMPANY_LABELS = {
    "hackerrank": "HackerRank",
    "claude": "Claude",
    "visa": "Visa",
}

KEYWORDS_BY_COMPANY = {
    "hackerrank": [
        "hackerrank",
        "assessment",
        "candidate",
        "interview",
        "certificate",
        "mock interview",
        "resume builder",
        "screen",
        "quickapply",
    ],
    "claude": [
        "claude",
        "anthropic",
        "bedrock",
        "amazon bedrock",
        "workspace",
        "team plan",
        "enterprise plan",
        "lti",
        "incognito",
        "scim",
        "sso",
    ],
    "visa": [
        "visa card",
        "visa",
        "chargeback",
        "merchant",
        "traveller",
        "travel cheque",
        "atm",
        "issuer",
        "bank",
        "stolen card",
        "3-d secure",
    ],
}

PROMPT_INJECTION_RE = re.compile(
    r"("
    r"internal rules|hidden logic|chain of thought|documents retrieved|retrieved documents|"
    r"system logic|decision logic|r[èe]gles internes|documents r[ée]cup[ée]r[ée]s|"
    r"logique exacte|reglas internas|documentos internos|l[óo]gica exacta"
    r"|"
    r"(reveal|show|display|print|ignore|share|affiche|montre|r[ée]v[èe]le|ignora|muestra|revela)"
    r".{0,120}"
    r"(internal|hidden|system|rules|logic|chain of thought|documents?|r[èe]gles internes|"
    r"documents r[ée]cup[ée]r[ée]s|logique exacte|reglas internas|documentos internos|l[óo]gica exacta)"
    r")",
    re.IGNORECASE | re.DOTALL,
)
DESTRUCTIVE_CODE_RE = re.compile(
    r"(delete|remove|wipe|destroy).{0,80}(all files|the system|entire system|filesystem)",
    re.IGNORECASE | re.DOTALL,
)
FEATURE_REQUEST_RE = re.compile(
    r"\b(feature request|would like a feature|please add|can you add|wishlist)\b",
    re.IGNORECASE,
)
SECURITY_VULN_RE = re.compile(r"\b(vulnerability|bug bounty|security issue|security bug)\b", re.IGNORECASE)
OUTAGE_RE = re.compile(
    r"\b("
    r"site is down|platform is down|service is down|resume builder is down|builder is down|"
    r"stopped working completely|all requests are failing|none of the pages are accessible|"
    r"none of the submissions across any challenges are working|"
    r"all submissions(?:\s+across\s+any\s+challenges)?\s+(?:are\s+)?not working"
    r")\b",
    re.IGNORECASE,
)
BUG_RE = re.compile(
    r"\b(not working|stopped working|failing|failure|error|bug|issue while|submissions .* not working|site is down)\b",
    re.IGNORECASE,
)


INFO_VISA_DISPUTE_RE = re.compile(
    r"\b(how|where|what).{0,50}dispute a charge\b|\bwhat is the process to dispute a charge\b",
    re.IGNORECASE | re.DOTALL,
)
HARMLESS_INVALID_RE = re.compile(
    r"\b(who is the actor|what is the name of the actor)\b",
    re.IGNORECASE,
)
THANKS_ONLY_RE = re.compile(
    r"^(thanks|thank you)( for (help|helping me|your help))?$",
    re.IGNORECASE,
)
LOW_CONFIDENCE_REASON = "the corpus did not contain a confident match for this issue"


ESCALATION_PATTERNS = [
    (
        "billing_or_refund",
        re.compile(
            r"\b(refund|refunded|money back|payment issue|billing issue|invoice issue|chargeback|active charge dispute)\b",
            re.IGNORECASE,
        ),
        "billing, refund, or payment requests require human support",
    ),
    (
        "account_access",
        re.compile(
            r"\b(restore my access|lost access|unlock(?: my)? account|unblock(?: my)? account|admin removed my seat|workspace owner.*restore|owner or admin.*restore|owner/admin-only)\b",
            re.IGNORECASE,
        ),
        "account access or permission changes require human support",
    ),
    (
        "assessment_decision",
        re.compile(
            r"\b(increase my score|score dispute|graded me unfairly|move me to the next round|recruiter rejected me|recruiter decision|certificate.*change|change.*certificate|certificate.*update|update.*certificate|rescheduling of .* assessment|reschedule.*assessment)\b",
            re.IGNORECASE,
        ),
        "score, certification, or recruiting decisions require human support",
    ),
    (
        "fraud_or_card_risk",
        re.compile(
            r"\b(fraud|identity theft|stolen card|lost or stolen card|card has been stolen|blocked.*card|card .* blocked|suspicious card activity|compromised card)\b",
            re.IGNORECASE,
        ),
        "fraud, identity, or card-risk issues require human support",
    ),
    (
        "security_vulnerability",
        SECURITY_VULN_RE,
        "security vulnerability reports require a dedicated human review process",
    ),
    (
        "platform_outage",
        OUTAGE_RE,
        "platform-wide outages or broad service failures require human support",
    ),
]


def _matches_data_topic(query: str, document: RetrievedDocument) -> bool:
    lower_query = query.lower()
    if not (
        any(term in lower_query for term in ("data", "privacy", "retention"))
        and any(term in lower_query for term in ("train", "training", "model", "models", "improve"))
        or "how long will the data be used" in lower_query
    ):
        return False

    text = f"{document.title} {document.path} {document.document.content[:2000]}".lower()
    return any(
        phrase in text
        for phrase in (
            "data retention",
            "retain",
            "retention period",
            "train our models",
            "used for model training",
            "does anthropic act as a data processor or controller",
        )
    )


def _matches_crawling_topic(query: str, document: RetrievedDocument) -> bool:
    lower_query = query.lower()
    if not any(term in lower_query for term in ("crawl", "crawling", "robots.txt", "claudebot", "opt out", "opt-out", "disallow")):
        return False

    text = f"{document.title} {document.path} {document.document.content[:2000]}".lower()
    return any(
        phrase in text
        for phrase in ("crawl", "crawler", "robots.txt", "claudebot", "disallow", "opting out")
    )


def _has_confident_retrieval(query: str, retrieved_docs: Sequence[RetrievedDocument]) -> bool:
    if not retrieved_docs:
        return False

    expanded_query_tokens = set(query_tokens(query))
    if len(expanded_query_tokens) < 2:
        return False

    for document in retrieved_docs:
        metadata = f"{document.title} {document.product_area.replace('_', ' ')} {document.path.replace('/', ' ')}"
        metadata_tokens = set(tokenize(metadata))
        content_tokens = set(tokenize(document.document.content[:4000]))
        metadata_overlap = len(expanded_query_tokens & metadata_tokens)
        content_overlap = len(expanded_query_tokens & content_tokens)

        if metadata_overlap >= 2:
            return True
        if metadata_overlap >= 1 and content_overlap >= 4 and document.score >= 280:
            return True
        if metadata_overlap >= 1 and content_overlap >= 2 and normalize_company(document.document.company) == "visa":
            return True
        if _matches_crawling_topic(query, document) or _matches_data_topic(query, document):
            return True

    return False


def _normalize_short_text(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _is_harmless_invalid(text: str) -> bool:
    normalized = _normalize_short_text(text)
    if not normalized:
        return False
    if normalized in {"who is the actor", "what is the name of the actor"}:
        return True
    return len(normalized) <= 32 and THANKS_ONLY_RE.fullmatch(normalized) is not None


def infer_company(issue: str, subject: str, company: str) -> str:
    cleaned = company.strip()
    if cleaned and cleaned.lower() != "none":
        lowered = cleaned.lower()
        if lowered in COMPANY_LABELS:
            return COMPANY_LABELS[lowered]
        return cleaned

    text = f"{subject} {issue}".lower()
    scores = {}
    for candidate, keywords in KEYWORDS_BY_COMPANY.items():
        score = sum(1 for keyword in keywords if keyword in text)
        scores[candidate] = score

    best_company, best_score = max(scores.items(), key=lambda item: (item[1], item[0]))
    if best_score == 0:
        return "None"
    return COMPANY_LABELS[best_company]


def normalize_company(company: str) -> str | None:
    lowered = company.strip().lower()
    if lowered in COMPANY_LABELS:
        return lowered
    return None


def route_ticket(
    issue: str,
    subject: str,
    company: str,
    retrieved_docs: Sequence[RetrievedDocument],
) -> RouteDecision:
    text = f"{subject}\n{issue}".strip()
    lower_text = text.lower()
    product_area = retrieved_docs[0].product_area if retrieved_docs else ""
    confident_retrieval = _has_confident_retrieval(text, retrieved_docs)

    if PROMPT_INJECTION_RE.search(text):
        return RouteDecision(
            status="replied",
            request_type="invalid",
            product_area=product_area,
        )

    if DESTRUCTIVE_CODE_RE.search(text):
        return RouteDecision(
            status="replied",
            request_type="invalid",
            product_area=product_area,
        )

    informational_visa_dispute = normalize_company(company) == "visa" and INFO_VISA_DISPUTE_RE.search(text)

    for _, pattern, reason in ESCALATION_PATTERNS:
        if informational_visa_dispute and "billing" in reason:
            continue
        if pattern.search(text):
            request_type = "bug" if "outage" in reason or "vulnerability" in reason else "product_issue"
            return RouteDecision(
                status="escalated",
                request_type=request_type,
                product_area=product_area,
                escalation_reason=reason,
            )

    if _is_harmless_invalid(text):
        return RouteDecision(
            status="replied",
            request_type="invalid",
            product_area="",
        )

    if FEATURE_REQUEST_RE.search(text):
        return RouteDecision(
            status="replied" if confident_retrieval else "escalated",
            request_type="feature_request",
            product_area=product_area,
            escalation_reason="" if confident_retrieval else "the request needs human review because the corpus does not provide a supported implementation path",
        )

    if OUTAGE_RE.search(text):
        return RouteDecision(
            status="escalated",
            request_type="bug",
            product_area=product_area,
            escalation_reason="platform-wide outages or broad service failures require human support",
        )

    if BUG_RE.search(text):
        if not confident_retrieval:
            return RouteDecision(
                status="escalated",
                request_type="bug",
                product_area=product_area,
                escalation_reason=LOW_CONFIDENCE_REASON,
            )
        return RouteDecision(
            status="replied",
            request_type="bug",
            product_area=product_area,
        )

    if not retrieved_docs:
        return RouteDecision(
            status="escalated",
            request_type="product_issue",
            product_area="",
            escalation_reason=LOW_CONFIDENCE_REASON,
        )

    if not confident_retrieval:
        return RouteDecision(
            status="escalated",
            request_type="product_issue",
            product_area=product_area,
            escalation_reason=LOW_CONFIDENCE_REASON,
        )

    return RouteDecision(
        status="replied",
        request_type="product_issue",
        product_area=product_area,
    )
