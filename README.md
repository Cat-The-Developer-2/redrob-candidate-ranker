# Redrob Candidate Ranker

Ranks candidates from candidates.jsonl against the Senior AI Engineer JD using:
1. Semantic similarity (sentence-transformers embeddings, precomputed)
2. Rule-based scoring (title relevance, skills, experience, location)
3. Behavioral signal multipliers (activity, responsiveness, notice period)
4. Honeypot/trap filtering

## Reproduce
```
pip install -r requirements.txt

python rank.py --candidates ./candidates.jsonl --out ./submission.csv

```


Embeddings (`candidate_embeddings.npy`, `jd_embedding.npy`) are precomputed and included — generated via `all-MiniLM-L6-v2` on GPU (Colab), but the ranking step itself runs CPU-only in ~15 seconds.
