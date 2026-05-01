from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, List


@dataclass(frozen=True)
class Document:
    company: str
    product_area: str
    title: str
    path: str
    content: str


COMPANIES = {"hackerrank", "claude", "visa"}

PRODUCT_AREA_ALIASES = {
    "hackerrank_community": "community",
    "conversation_management": "conversation_management",
    "travel_support": "travel_support",
    "privacy_and_legal": "privacy",
    "team_and_enterprise_plans": "team_and_enterprise",
    "amazon_bedrock": "amazon_bedrock",
    "claude_for_education": "claude_for_education",
    "claude_for_government": "claude_for_government",
    "claude_for_nonprofits": "claude_for_nonprofits",
    "identity_management_sso_jit_scim": "identity_management",
    "pro_and_max_plans": "pro_and_max_plans",
    "claude_api_and_console": "api_and_console",
    "claude_mobile_apps": "mobile_apps",
    "privacy_and_legal": "privacy",
    "general_help": "general_help",
}

IGNORED_SEGMENTS = {
    "support",
    "consumer",
    "merchant",
    "root",
}


def normalize_segment(segment: str) -> str:
    normalized = segment.strip().lower()
    normalized = normalized.replace("&", "and")
    normalized = normalized.replace("-", "_")
    normalized = re.sub(r"[^a-z0-9_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return PRODUCT_AREA_ALIASES.get(normalized, normalized)


def infer_product_area(company: str, relative_path: Path) -> str:
    parts = [normalize_segment(part) for part in relative_path.parts[1:-1]]
    stem = normalize_segment(relative_path.stem)

    if company == "visa":
        visa_parts = parts + [stem]
        if "travel_support" in visa_parts:
            return "travel_support"
        if "small_business" in visa_parts:
            return "small_business"
        if any(part in {"travelers_cheques", "travelers_cheques_for_consumers"} for part in visa_parts):
            return "travel_support"
        return "general_support"

    for part in parts:
        if part and part not in IGNORED_SEGMENTS and part not in {company, "index"}:
            return PRODUCT_AREA_ALIASES.get(part, part)

    filename_area = stem
    if filename_area == "index":
        return "general_support"
    return filename_area or "general_support"


def infer_title(file_path: Path, content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()

    fallback = file_path.stem
    fallback = re.sub(r"^\d+-", "", fallback)
    fallback = fallback.replace("-", " ").replace("_", " ").strip()
    return fallback.title() or file_path.name


def iter_markdown_files(data_root: Path) -> Iterable[Path]:
    for path in sorted(data_root.rglob("*.md")):
        if path.is_file():
            yield path


def load_corpus(data_root: str | Path) -> List[Document]:
    root = Path(data_root).resolve()
    documents: List[Document] = []

    for file_path in iter_markdown_files(root):
        relative_path = file_path.relative_to(root)
        if not relative_path.parts:
            continue

        company = normalize_segment(relative_path.parts[0])
        if company not in COMPANIES:
            continue

        content = file_path.read_text(encoding="utf-8")
        title = infer_title(file_path, content)
        product_area = infer_product_area(company, relative_path)

        documents.append(
            Document(
                company=company,
                product_area=product_area,
                title=title,
                path=relative_path.as_posix(),
                content=content,
            )
        )

    return documents
