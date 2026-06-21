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

Candidate profiles and the JD are embedded using [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Cosine similarity between each candidate's embedding and the JD embedding captures semantic fit — surfacing candidates with genuinely relevant experience even
