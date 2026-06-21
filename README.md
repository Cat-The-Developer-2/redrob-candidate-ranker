# Redrob Intelligent Candidate Discovery & Ranking

**Hackathon submission** · Team: Kitty-The-Developer · Solo participant: Aditya Chaurasia

A hybrid candidate ranker that surfaces the **top 100 Senior AI Engineers** from a 100,000-candidate pool against a given Job Description — without relying on keyword matching alone.

---

## Overview

Pure keyword search misses candidates whose profiles are semantically relevant but don't share exact vocabulary with the JD. Pure embedding similarity, on the other hand, is easily fooled by keyword-stuffed or tangentially related profiles (during testing, HR Managers who mentioned "GenAI" in passing ranked near the top on cosine similarity alone).

This ranker combines **semantic embeddings** with **explicit rule-based signals** and **behavioral multipliers** to produce a score that reflects genuine fit — not surface-level overlap.

---

## How It Works

### 1 · Semantic Similarity (30% of base score)

Candidate profiles and the JD are embedded using [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Cosine similarity between each candidate's embedding and the JD embedding captures semantic fit — surfacing candidates with genuinely relevant experience even when they use different terminology.

Embeddings are **precomputed once** (see [Two-Phase Compute](#two-phase-compute) below) and loaded from `.npy` files at ranking time.

### 2 · Rule-Based Scoring (70% of base score)

| Signal | Weight | Logic |
|---|---|---|
| **Title relevance** | 25% | Matches current and historical titles against AI/ML/Search keywords |
| **Key skills** | 20% | Vector DBs (Pinecone, FAISS, Weaviate, Milvus, …) · Retrieval (RAG, embeddings, IR) · Eval frameworks (NDCG, MRR, A/B) |
| **Experience fit** | 15% | Full score for 5–9 years; partial credit for 3–5 or 9–12; minimal otherwise |
| **Location fit** | 10% | Full score for Pune / Noida / Hyderabad / Mumbai / Delhi NCR |

### 3 · Penalties (multiplicative)

Applied to the base score before behavioral signals:

- **Consulting-only career** (`×0.4`): All career history at TCS / Infosys / Wipro / Accenture / Cognizant / Capgemini with no product company exposure.
- **Title chaser** (`×0.6`): Majority of job stints shorter than 18 months — signals instability over genuine progression.
- **Framework enthusiast** (`×0.5`): LangChain listed but no evidence of actual vector DB, retrieval, or evaluation work — surface-level AI exposure without production depth.

### 4 · Behavioral Signal Multiplier

Scores are further scaled by `redrob_signals` platform data:

| Signal | Effect |
|---|---|
| `open_to_work_flag = false` | `×0.7` |
| `recruiter_response_rate < 20%` | `×0.7` |
| `last_active_date` > 180 days ago | `×0.5` |
| `last_active_date` 90–180 days ago | `×0.8` |
| `notice_period_days > 60` | `×0.85` |

A high paper score means nothing if the candidate is unreachable or passive. Multipliers compound — a candidate who is inactive *and* unresponsive gets down-weighted significantly.

### 5 · Honeypot Filtering

Candidates with **impossible or inconsistent profiles** are excluded outright (final score set to `-1.0`):

- `proficiency = "expert"` with `duration_months = 0` for any skill.
- `years_of_experience * 12` exceeds total career history duration by more than 24 months.

These profiles are treated as synthetic noise injected to test the ranker's integrity.

---

## Two-Phase Compute

The solution separates the expensive step from the fast step:

| Phase | Where | Time | GPU? | Network? |
|---|---|---|---|---|
| **Embedding pre-computation** | Google Colab (T4 GPU) | ~6–7 min | ✅ Yes | ✅ Yes (model download once) |
| **Ranking** | Any CPU machine | ~13–15 sec | ❌ No | ❌ No |

**Why this satisfies the constraint:** The competition specifies that the *ranking step* must run within 5 minutes on ≤16 GB RAM without a GPU or network access. `rank.py` only loads precomputed `.npy` files and runs pure NumPy / scikit-learn operations on CPU — no model inference, no API calls. The one-time embedding generation (which does need GPU) is a setup step committed to the repo, not part of the judged execution path.

The precomputed embeddings (`float16`, ~73 MB total) are committed to the repository so judges can reproduce results without re-running the Colab step.

---

## Setup

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `scikit-learn`, `pandas`, `sentence-transformers`

---

## Reproduce

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Optional flags (defaults shown):
```bash
python rank.py \
  --candidates ./candidates.jsonl \
  --out ./submission.csv \
  --embeddings ./candidate_embeddings.npy \
  --jd_embedding ./jd_embedding.npy
```

The script accepts gzip-compressed JSONL (`.jsonl.gz`) transparently.

Expected output:
