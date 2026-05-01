from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from corpus import load_corpus
from response_builder import build_response
from retriever import LexicalRetriever
from router import infer_company, normalize_company, route_ticket


OUTPUT_COLUMNS = [
    "issue",
    "subject",
    "company",
    "response",
    "product_area",
    "status",
    "request_type",
    "justification",
]


def trim(value: str | None) -> str:
    return (value or "").strip()


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Deterministic support triage agent")
    parser.add_argument(
        "--input",
        default=str(repo_root / "support_tickets" / "support_tickets.csv"),
        help="Path to the input CSV",
    )
    parser.add_argument(
        "--output",
        default=str(repo_root / "support_tickets" / "output.csv"),
        help="Path to the output CSV",
    )
    return parser


def read_rows(input_path: Path) -> List[Dict[str, str]]:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def process_rows(rows: List[Dict[str, str]]) -> tuple[List[Dict[str, str]], List[str]]:
    repo_root = Path(__file__).resolve().parents[1]
    documents = load_corpus(repo_root / "data")
    retriever = LexicalRetriever(documents)
    warnings: List[str] = []
    output_rows: List[Dict[str, str]] = []

    for index, row in enumerate(rows, start=1):
        issue = trim(row.get("Issue"))
        subject = trim(row.get("Subject"))
        raw_company = trim(row.get("Company"))
        resolved_company = infer_company(issue=issue, subject=subject, company=raw_company)
        company_key = normalize_company(resolved_company) if resolved_company != "None" else None

        query = " ".join(part for part in [subject, issue] if part)
        retrieved_docs = retriever.search(query=query, company=company_key, top_k=3)
        route = route_ticket(
            issue=issue,
            subject=subject,
            company=resolved_company,
            retrieved_docs=retrieved_docs,
        )
        built = build_response(
            issue=issue,
            subject=subject,
            company=resolved_company,
            route=route,
            retrieved_docs=retrieved_docs,
        )

        if not retrieved_docs:
            warnings.append(f"Row {index}: no supporting documents retrieved.")

        output_rows.append(
            {
                "issue": issue,
                "subject": subject,
                "company": resolved_company,
                "response": built.response,
                "product_area": route.product_area,
                "status": route.status,
                "request_type": route.request_type,
                "justification": built.justification,
            }
        )

    return output_rows, warnings


def write_rows(output_path: Path, rows: List[Dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    rows = read_rows(input_path)
    output_rows, warnings = process_rows(rows)
    write_rows(output_path, output_rows)

    print(f"Processed {len(output_rows)} tickets.")
    print(f"Wrote output to {output_path}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
