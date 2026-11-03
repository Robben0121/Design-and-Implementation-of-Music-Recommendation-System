"""
eval_metrics.py - Evaluation metrics for the music recommendation experiments.

Rule-based automatic evaluation:
- Each test query carries a `relevance` rule (genres / tags / keywords).
- A song's graded relevance (0-3) is computed from the rule:
    3 = genre match AND (tag match OR keyword match)
    2 = genre match only
    1 = tag match OR keyword match only
    0 = no match
- NDCG@k / Precision@k / Recall@k are computed against the full relevant set
  (all songs in the database with relevance > 0 for that query).
"""

import math
from typing import Dict, List, Optional, Tuple, Any

from data_loader import MusicDatabase, MusicItem


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def relevance_level(song: MusicItem, rule: Dict) -> int:
    """Graded relevance of a song for a query, using the query's relevance rule."""
    genres = {_norm(g) for g in rule.get("genres", [])}
    tags = {_norm(t) for t in rule.get("tags", [])}
    keywords = [_norm(k) for k in rule.get("keywords", [])]

    genre_match = _norm(song.genre) in genres
    tag_match = bool(tags) and any(_norm(t) in tags for t in song.tags)

    keyword_match = False
    if keywords:
        haystack = _norm(song.semantic_summary) + " " + _norm(song.description)
        keyword_match = any(kw in haystack for kw in keywords)

    if genre_match and (tag_match or keyword_match):
        return 3
    if genre_match:
        return 2
    if tag_match or keyword_match:
        return 1
    return 0


def build_relevant_set(music_db: MusicDatabase, rule: Dict) -> Dict[str, int]:
    """Return {song_id: graded_relevance} for every relevant song in the DB (rel >= 1)."""
    rel_map: Dict[str, int] = {}
    for item in music_db.music_items:
        rel = relevance_level(item, rule)
        if rel >= 1:
            rel_map[item.song_id] = rel
    return rel_map


def _gains(ranked_ids: List[Optional[str]], rel_map: Dict[str, int], k: int) -> List[int]:
    """Graded relevance gains for the first k ranked song_ids (None = hallucination -> 0)."""
    return [rel_map.get(sid, 0) if sid is not None else 0 for sid in ranked_ids[:k]]


def precision_at_k(ranked_ids: List[Optional[str]], rel_map: Dict[str, int], k: int = 5) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for g in _gains(ranked_ids, rel_map, k) if g > 0)
    return hits / k


def recall_at_k(ranked_ids: List[Optional[str]], rel_map: Dict[str, int], k: int = 5) -> float:
    if not rel_map:
        return 0.0
    hits = sum(1 for g in _gains(ranked_ids, rel_map, k) if g > 0)
    return hits / len(rel_map)


def ndcg_at_k(ranked_ids: List[Optional[str]], rel_map: Dict[str, int], k: int = 5) -> float:
    gains = _gains(ranked_ids, rel_map, k)

    dcg = 0.0
    for i, g in enumerate(gains):
        if g > 0:
            dcg += g / math.log2(i + 2)  # i is 0-based -> discount log2(i+2)

    # Ideal ranking: sort all relevant items' relevance descending, take top k.
    ideal = sorted((g for g in rel_map.values() if g > 0), reverse=True)[:k]
    idcg = 0.0
    for i, g in enumerate(ideal):
        idcg += g / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_query(ranked_ids: List[Optional[str]], rel_map: Dict[str, int], k: int = 5) -> Dict[str, float]:
    """Convenience: all three metrics for one query."""
    return {
        "ndcg@5": ndcg_at_k(ranked_ids, rel_map, k),
        "precision@5": precision_at_k(ranked_ids, rel_map, k),
        "recall@5": recall_at_k(ranked_ids, rel_map, k),
    }


if __name__ == "__main__":
    # Quick self-test with a tiny in-memory DB.
    from data_loader import MusicDatabase

    class _TmpDB:
        music_items = [
            MusicItem(song_id="a", song="A", artist="x", album="", genre="Pop", year=2020,
                      release_date="", album_type="", media="", description="calm study", tags=["Piano"], semantic_summary="relax"),
            MusicItem(song_id="b", song="B", artist="x", album="", genre="Pop", year=2020,
                      release_date="", album_type="", media="", description="", tags=["Catchy"], semantic_summary=""),
            MusicItem(song_id="c", song="C", artist="x", album="", genre="Rock", year=2020,
                      release_date="", album_type="", media="", description="", tags=["Guitar"], semantic_summary=""),
        ]

    rule = {"genres": ["Pop"], "tags": ["Piano"], "keywords": ["relax"]}
    rel_map = build_relevant_set(_TmpDB(), rule)
    print("rel_map:", rel_map)  # a -> 3, b -> 2 (genre match only), c -> 0
    print("evaluate [a, b, c, None, None]:", evaluate_query(["a", "b", "c", None, None], rel_map))
