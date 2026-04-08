---
title: Invoice Exception Openenv
emoji: crab
colorFrom: blue
colorTo: yellow
sdk: docker
pinned: false
---

# Invoice Exception OpenEnv

This project is a Round 1 OpenEnv submission for a real-world finance operations task: supplier invoice exception handling.

The environment exposes a standard `reset()` / `step()` / `state()` loop over a realistic workflow:

- review invoice evidence
- compare the invoice to the purchase order
- inspect goods receipt records
- search for duplicates
- request supporting documentation
- decide whether to approve, escalate, or reject

Each `reset()` call creates an isolated `session_id`, so multiple evaluation runs can happen without clobbering each other.

## Tasks

The environment includes four graded tasks:

1. `invoice_easy`: a standard 3-way match that should be approved.
2. `invoice_medium`: a price mismatch that should be routed to manual review.
3. `invoice_hard`: a likely duplicate invoice that should be rejected.
4. `invoice_expert`: an approved change-order exception that should be approved only after supporting evidence is collected.

Each task provides dense intermediate reward plus a final score in the `0.0` to `1.0` range. The grader rewards evidence gathering and correct resolution, not just one rigid action order.

## Files

- `app.py`: self-contained HTTP environment server
- `openenv.yaml`: API and model metadata
- `inference.py`: required root-level baseline script
- `smoke_test.py`: quick local verification
- `Dockerfile`: container for Hugging Face Spaces deployment
- `.env.example`: expected environment variables

## Local Run

Start the server:

```bash
python app.py
```

In another shell, run the baseline:

```bash
python inference.py
```

Run the smoke test:

```bash
python smoke_test.py
```

## Environment Variables

The hackathon dashboard calls out these variables:

- `API_BASE_URL`: base URL for OpenAI-compatible model calls
- `MODEL_NAME`: model identifier
- `HF_TOKEN`: authentication token for inference provider

The server itself does not need these variables. `inference.py` uses them when available and otherwise falls back to a deterministic baseline policy for local testing.

## Deployment

Build locally:

```bash
docker build -t invoice-exception-openenv .
docker run -p 7860:7860 invoice-exception-openenv
```

For Hugging Face Spaces, create a Docker Space and upload the repository contents as-is.

## Notes

- Runtime is intentionally lightweight and CPU-friendly.
- No external services are required for the environment server.
- The baseline emits `[START]`, `[STEP]`, and `[END]` logs in the format described on the dashboard.
- Session-based episode state makes the environment safer for concurrent validation runs.
