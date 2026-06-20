#!/usr/bin/env python3
"""
Redrob Hackathon — Candidate Ranker
Usage: python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import argparse
import json
import gzip
import re
from datetime import date
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

CONSULTING_FIRMS = {'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini'}
PRODUCT_LOCATIONS = {'pune', 'noida', 'hyderabad', 'mumbai', 'delhi', 'gurgaon', 'gurugram', 'bengaluru', 'bangalore'}
RELEVANT_TITLE_KEYWORDS = ['ai engineer', 'ml engineer', 'machine learning', 'data scientist',
                            'nlp', 'search', 'ranking', 'recommendation', 'applied scientist',
                            'backend engineer', 'data engineer', 'software engineer']
VECTOR_DB_SKILLS = {'elasticsearch', 'faiss', 'milvus', 'opensearch', 'pinecone',
                     'qdrant', 'weaviate', 'pgvector', 'vector search'}
RETRIEVAL_SKILLS = {'embeddings', 'information retrieval', 'rag', 'vector search'}
EVAL_SKILLS = {'ndcg', 'mrr', 'map', 'a/b testing', 'evaluation', 'learning to rank'}
FRAMEWORK_ONLY_SKILL = {'langchain'}
TODAY = date(2026, 6, 20)


def load_candidates(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        return [json.loads(line) for line in f if line.strip()]


def title_relevance_score(c):
    titles = [c['profile']['current_title'].lower()]
    titles += [job['title'].lower() for job in c.get('career_history', [])]
    matches = sum(1 for t in titles for kw in RELEVANT_TITLE_KEYWORDS if kw in t)
    return min(matches / 3, 1.0)


def is_consulting_only(c):
    companies = [job['company'].lower() for job in c.get('career_history', [])]
    if not companies:
        return False
    return all(any(firm in comp for firm in CONSULTING_FIRMS) for comp in companies)


def is_title_chaser(c):
    history = c.get('career_history', [])
    if len(history) < 3:
        return False
    short_stints = sum(1 for job in history if job.get('duration_months', 999) < 18)
    return short_stints >= len(history) - 1


def location_fit(c):
    loc = c['profile']['location'].lower()
    return any(city in loc for city in PRODUCT_LOCATIONS)


def experience_fit(c):
    yoe = c['profile']['years_of_experience']
    if 5 <= yoe <= 9:
        return 1.0
    elif 3 <= yoe < 5 or 9 < yoe <= 12:
        return 0.6
    return 0.2


def skills_score(c):
    skill_names = {s['name'].lower() for s in c.get('skills', [])}
    score = 0.0
    if skill_names & VECTOR_DB_SKILLS:
        score += 0.4
    if skill_names & RETRIEVAL_SKILLS:
        score += 0.4
    if skill_names & EVAL_SKILLS:
        score += 0.2
    return min(score, 1.0)


def is_framework_enthusiast(c):
    skills = c.get('skills', [])
    skill_names = {s['name'].lower() for s in skills}
    has_only_langchain = (skill_names & FRAMEWORK_ONLY_SKILL) and not (
        skill_names & (VECTOR_DB_SKILLS | RETRIEVAL_SKILLS | EVAL_SKILLS))
    for s in skills:
        if s['name'].lower() in FRAMEWORK_ONLY_SKILL and s.get('duration_months', 99) < 12:
            return True
    return has_only_langchain


def honeypot_flag(c):
    for skill in c.get('skills', []):
        if skill.get('proficiency') == 'expert' and skill.get('duration_months', 0) == 0:
            return True
    total_months = sum(job.get('duration_months', 0) for job in c.get('career_history', []))
    if c['profile']['years_of_experience'] * 12 > total_months + 24:
        return True
    return False


def signal_multiplier(c):
    s = c['redrob_signals']
    mult = 1.0
    if not s.get('open_to_work_flag', True):
        mult *= 0.7
    if s.get('recruiter_response_rate', 1.0) < 0.2:
        mult *= 0.7
    try:
        last_active = date.fromisoformat(s['last_active_date'])
        days_inactive = (TODAY - last_active).days
        if days_inactive > 180:
            mult *= 0.5
        elif days_inactive > 90:
            mult *= 0.8
    except Exception:
        pass
    if s.get('notice_period_days', 0) > 60:
        mult *= 0.85
    return mult


def build_candidate_text(c):
    profile = c['profile']
    parts = [
        profile.get('headline', ''),
        profile.get('summary', ''),
        f"Current role: {profile.get('current_title','')} at {profile.get('current_company','')}",
    ]
    for job in c.get('career_history', []):
        parts.append(f"{job.get('title','')} at {job.get('company','')}: {job.get('description','')}")
    skill_names = [s['name'] for s in c.get('skills', [])]
    if skill_names:
        parts.append("Skills: " + ", ".join(skill_names))
    return " | ".join(p for p in parts if p)


def generate_reasoning(c, score_info):
    p = c['profile']
    s = c['redrob_signals']
    parts = [f"{p['current_title']} at {p['current_company']} with {p['years_of_experience']:.1f} years experience."]

    skill_names = [sk['name'] for sk in c.get('skills', [])]
    relevant_found = [sk for sk in skill_names if sk.lower() in
                       (VECTOR_DB_SKILLS | RETRIEVAL_SKILLS | EVAL_SKILLS)]
    if relevant_found:
        parts.append(f"Has hands-on {', '.join(relevant_found[:3])} experience.")

    if location_fit(c):
        parts.append(f"Based in {p['location']}, matching preferred locations.")
    else:
        parts.append(f"Located in {p['location']} (outside preferred cities).")

    if score_info['sig_mult'] < 0.8:
        concerns = []
        if not s.get('open_to_work_flag', True):
            concerns.append("not marked open to work")
        if s.get('recruiter_response_rate', 1) < 0.2:
            concerns.append(f"low recruiter response rate ({s.get('recruiter_response_rate',0):.0%})")
        if s.get('notice_period_days', 0) > 60:
            concerns.append(f"{s.get('notice_period_days')}-day notice period")
        if concerns:
            parts.append(f"Some concerns: {', '.join(concerns)}.")
    else:
        parts.append(f"Active on platform with {s.get('recruiter_response_rate',0):.0%} recruiter response rate.")

    if score_info['consulting_penalty'] < 1.0:
        parts.append("Entire career at consulting firms — limited product company exposure.")

    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--embeddings', default='candidate_embeddings.npy',
                         help='Path to precomputed candidate embeddings')
    parser.add_argument('--jd_embedding', default='jd_embedding.npy',
                         help='Path to precomputed JD embedding')
    args = parser.parse_args()

    print("Loading candidates...")
    candidates = load_candidates(args.candidates)

    print("Loading precomputed embeddings...")
    candidate_embeddings = np.load(args.embeddings)
    jd_embedding = np.load(args.jd_embedding)

    similarity_scores = cosine_similarity(jd_embedding, candidate_embeddings)[0]
    sim_min, sim_max = similarity_scores.min(), similarity_scores.max()
    semantic_norm = (similarity_scores - sim_min) / (sim_max - sim_min)

    print("Scoring candidates...")
    results = []
    for i, c in enumerate(candidates):
        semantic = semantic_norm[i]
        title_score = title_relevance_score(c)
        exp_score = experience_fit(c)
        loc_score = 1.0 if location_fit(c) else 0.3
        sk_score = skills_score(c)
        consulting_penalty = 0.4 if is_consulting_only(c) else 1.0
        chaser_penalty = 0.6 if is_title_chaser(c) else 1.0
        framework_penalty = 0.5 if is_framework_enthusiast(c) else 1.0
        sig_mult = signal_multiplier(c)
        honeypot = honeypot_flag(c)

        base_score = (0.30 * semantic + 0.25 * title_score + 0.20 * sk_score +
                      0.15 * exp_score + 0.10 * loc_score)
        final_score = base_score * consulting_penalty * chaser_penalty * framework_penalty * sig_mult

        if honeypot:
            final_score = -1.0

        results.append({
            'candidate_id': c['candidate_id'], 'final_score': final_score,
            'consulting_penalty': consulting_penalty, 'sig_mult': sig_mult, 'idx': i
        })

    top_100 = sorted(results, key=lambda r: -r['final_score'])[:100]
    top_100_sorted = sorted(top_100, key=lambda r: (-round(r['final_score'], 4), candidates[r['idx']]['candidate_id']))

    print("Writing output...")
    import csv
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for rank, r in enumerate(top_100_sorted, start=1):
            c = candidates[r['idx']]
            reasoning = generate_reasoning(c, r)
            writer.writerow([c['candidate_id'], rank, round(r['final_score'], 4), reasoning])

    print(f"Done. Wrote {len(top_100_sorted)} rows to {args.out}")


if __name__ == '__main__':
    main()
