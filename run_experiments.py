"""
run_experiments.py - Run the recommendation experiments and compute metrics.

Compares 4 model variants on the fixed test set (data/test_queries.json):
    baseline1_vector     : ChromaDB semantic retrieval only (no LLM rerank)
    baseline2_llm        : pure LLM recommendation (no RAG)
    full                 : RAG retrieval + LLM rerank + user profile (our system)
    ablation_noprofile   : RAG retrieval + LLM rerank, profile disabled

Ground truth is rule-based (per-query genre/tag relevance rules); metrics are
NDCG@5 / Precision@5 / Recall@5 (see eval_metrics.py).

LLM outputs are cached under experiments/cache/ so re-runs are idempotent.
"""

import json
import os
from typing import List, Optional, Dict

from config import settings
from data_loader import MusicDatabase
from vector_engine import VectorSearchEngine
from llm_engine import LLMRecommendationEngine
from user_profile import UserProfile
from eval_metrics import build_relevant_set, evaluate_query

TOP_K = 5
CACHE_DIR = "./experiments/cache"
OUT_DIR = "./experiments"

MODELS = ["baseline1_vector", "baseline2_llm", "full", "ablation_noprofile"]


def init_backend():
    music_db = MusicDatabase(settings.MUSIC_DATA_PATH)
    vector_engine = VectorSearchEngine(music_db)
    vector_engine.build_index()
    llm_engine = LLMRecommendationEngine(vector_engine)
    return music_db, vector_engine, llm_engine


def build_profile_from_query(llm_engine: LLMRecommendationEngine, query: dict) -> UserProfile:
    """Faithful profile: extract preferences with the LLM, apply the same
    confidence-gated update the real app uses. Profiles land in a scratch dir
    so they don't pollute the real user_profiles store."""
    prefs = llm_engine.extract_user_preferences(query["text"])
    profile = UserProfile(f"exp_{query['id']}", "experiment",
                          profile_dir="./experiments/_profiles")
    if prefs:
        profile.update_preferences_from_text(prefs)
    return profile


def run_model(model: str, query: dict, backend, cache: dict) -> List[Optional[str]]:
    """Return an ordered list of song_ids (None = hallucinated slot) of length <= TOP_K."""
    _, vector_engine, llm_engine = backend
    qid = query["id"]
    qtext = query["text"]

    cache_key = f"{model}__{qid}"
    if cache_key in cache:
        return cache[cache_key]

    ranked: List[Optional[str]] = []

    if model == "baseline1_vector":
        results = vector_engine.search(qtext, top_k=TOP_K)
        ranked = [r["item"].song_id for r in results]

    elif model == "baseline2_llm":
        out = llm_engine.recommend_songs_llm_only(qtext, top_k=TOP_K)
        ranked = [item.song_id if item is not None else None for item in out["songs"]]
        # also stash hallucination info for the analysis
        cache[f"{model}__{qid}__meta"] = {
            "hallucinated": out["hallucinated"],
            "raw": out["raw"],
        }

    elif model == "full":
        profile = build_profile_from_query(llm_engine, query)
        out = llm_engine.recommend_songs(qtext, user_profile=profile, top_k=TOP_K)
        ranked = [item.song_id for item in out["recommendations"]]

    elif model == "ablation_noprofile":
        out = llm_engine.recommend_songs(qtext, user_profile=None, top_k=TOP_K)
        ranked = [item.song_id for item in out["recommendations"]]

    else:
        raise ValueError(f"unknown model: {model}")

    cache[cache_key] = ranked
    return ranked


def load_cache() -> dict:
    cache: Dict[str, object] = {}
    if not os.path.isdir(CACHE_DIR):
        return cache
    for fn in os.listdir(CACHE_DIR):
        if fn.endswith(".json"):
            key = fn[:-5]
            with open(os.path.join(CACHE_DIR, fn), encoding="utf-8") as f:
                cache[key] = json.load(f)
    return cache


def save_cache(cache: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    for key, val in cache.items():
        with open(os.path.join(CACHE_DIR, f"{key}.json"), "w", encoding="utf-8") as f:
            json.dump(val, f, ensure_ascii=False)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Initializing backend (DB + vector index + LLM engine)...")
    music_db, vector_engine, llm_engine = init_backend()
    backend = (music_db, vector_engine, llm_engine)

    queries = json.load(open("./data/test_queries.json", encoding="utf-8"))
    cache = load_cache()

    # per-query metrics: {model: {qid: {ndcg@5, precision@5, recall@5}}}
    metrics = {m: {} for m in MODELS}
    predictions = {}  # qid -> {model: [{"title","genre","rel"}]}
    hallucination = {}  # qid -> count for baseline2

    print(f"Running {len(queries)} queries x {len(MODELS)} models...\n")
    for qi, query in enumerate(queries, 1):
        qid = query["id"]
        rel_map = build_relevant_set(music_db, query["relevance"])
        predictions[qid] = {"query": query["text"], "models": {}}

        for model in MODELS:
            ranked = run_model(model, query, backend, cache)
            # cap to TOP_K (padded with None if short)
            ranked = (ranked + [None] * TOP_K)[:TOP_K]
            metrics[model][qid] = evaluate_query(ranked, rel_map, TOP_K)

            preds = []
            for sid in ranked:
                if sid is None:
                    preds.append({"song_id": None, "title": None, "genre": None, "rel": 0})
                    continue
                item = music_db.get_by_id(sid)
                preds.append({
                    "song_id": sid,
                    "title": item.song if item else "?",
                    "artist": item.artist if item else "?",
                    "genre": item.genre if item else "?",
                    "rel": rel_map.get(sid, 0),
                })
            predictions[qid]["models"][model] = preds

            if model == "baseline2_llm":
                meta = cache.get(f"{model}__{qid}__meta", {})
                hallucination[qid] = meta.get("hallucinated", None)

        print(f"[{qi}/{len(queries)}] {qid:10s} done")

    save_cache(cache)

    # ---- aggregate ----
    agg = {}
    for model in MODELS:
        m = metrics[model]
        agg[model] = {
            "ndcg@5": sum(v["ndcg@5"] for v in m.values()) / len(m),
            "precision@5": sum(v["precision@5"] for v in m.values()) / len(m),
            "recall@5": sum(v["recall@5"] for v in m.values()) / len(m),
        }

    # category breakdown
    categories = {}
    for model in MODELS:
        categories[model] = {}
        for q in queries:
            cat = q["category"]
            categories[model].setdefault(cat, []).append(metrics[model][q["id"]])
        for cat, vals in categories[model].items():
            categories[model][cat] = {
                "ndcg@5": sum(v["ndcg@5"] for v in vals) / len(vals),
                "precision@5": sum(v["precision@5"] for v in vals) / len(vals),
                "recall@5": sum(v["recall@5"] for v in vals) / len(vals),
            }

    # ---- write outputs ----
    with open(os.path.join(OUT_DIR, "results_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "overall": agg,
            "per_category": categories,
            "per_query": metrics,
            "hallucination": hallucination,
        }, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    write_markdown_table(agg, categories, hallucination, len(queries))

    # ---- print summary ----
    print("\n========== OVERALL METRICS ==========")
    header = f"{'model':22s} {'NDCG@5':>8s} {'P@5':>8s} {'R@5':>8s}"
    print(header)
    print("-" * len(header))
    for model in MODELS:
        a = agg[model]
        print(f"{model:22s} {a['ndcg@5']:8.4f} {a['precision@5']:8.4f} {a['recall@5']:8.4f}")
    print("\nResults written to experiments/")


def write_markdown_table(agg, categories, hallucination, n_queries):
    lines = []
    lines.append("# Experiment Results\n")
    lines.append(f"Test set: {n_queries} queries (data/test_queries.json) | top-k = {TOP_K}\n")
    lines.append("Ground truth: rule-based genre/tag relevance (see eval_metrics.py).\n")

    lines.append("## Overall metrics (averaged over all queries)\n")
    lines.append("| Model | NDCG@5 | Precision@5 | Recall@5 |")
    lines.append("|---|---|---|---|")
    for model in MODELS:
        a = agg[model]
        lines.append(f"| {model} | {a['ndcg@5']:.4f} | {a['precision@5']:.4f} | {a['recall@5']:.4f} |")

    lines.append("\n## Per-category Precision@5 / NDCG@5\n")
    lines.append("| Model | scene P@5 / NDCG | mood P@5 / NDCG | genre P@5 / NDCG | ambiguous P@5 / NDCG |")
    lines.append("|---|---|---|---|---|")
    order = ["scene", "mood", "genre", "ambiguous"]
    for model in MODELS:
        c = categories[model]
        cells = []
        for cat in order:
            if cat in c:
                cells.append(f"{c[cat]['precision@5']:.4f} / {c[cat]['ndcg@5']:.4f}")
            else:
                cells.append("-")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")

    if hallucination:
        lines.append("\n## Baseline-2 hallucination rate (pure LLM, no RAG)\n")
        lines.append("| Query | Unmatched (hallucinated) songs out of 5 |")
        lines.append("|---|---|")
        for qid, n in hallucination.items():
            lines.append(f"| {qid} | {n} |")

    lines.append("")
    with open(os.path.join(OUT_DIR, "results_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
