# Deterministic Support Triage Agent

## Run

From the repo root:

```bash
python3 code/main.py
```

Optional arguments:

```bash
python3 code/main.py --input support_tickets/support_tickets.csv --output support_tickets/output.csv
```

The agent writes:

- `support_tickets/output.csv`

## Architecture

The solution is intentionally small and deterministic:

- `corpus.py`
  Loads every markdown file under `data/`, infers `company`, `product_area`, `title`, and stores the raw content.
- `retriever.py`
  Builds a standard-library lexical index and ranks documents with token overlap plus TF-IDF style weighting.
- `router.py`
  Applies explicit routing rules for escalation, invalid requests, bugs, and company inference.
- `response_builder.py`
  Builds short grounded replies from retrieved sentences or concise escalation responses with safe next steps.
- `main.py`
  Reads the input CSV, runs retrieval and routing for each ticket, and writes the required output schema.

## Why deterministic retrieval and explicit rules

This challenge requires safe behavior, no web access, and strong escalation discipline. A deterministic pipeline is a good fit because it:

- stays grounded in the local corpus
- is reproducible without API keys
- makes escalation decisions auditable
- avoids hallucinated actions or unsupported policies

Explicit rules are especially useful for high-risk classes like billing, refunds, score disputes, account restoration, fraud, and security reports.

## Limitations

- Retrieval is lexical only, so paraphrased tickets may retrieve weaker evidence than a semantic system.
- `product_area` inference is path-based and intentionally coarse.
- Responses are extractive and concise rather than fully natural.
- Ambiguous tickets default toward escalation when the corpus does not support a safe direct answer.
