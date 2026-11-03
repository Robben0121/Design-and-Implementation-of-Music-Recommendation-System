# Experiment Results

Test set: 40 queries (data/test_queries.json) | top-k = 5

Ground truth: rule-based genre/tag relevance (see eval_metrics.py).

## Overall metrics (averaged over all queries)

| Model | NDCG@5 | Precision@5 | Recall@5 |
|---|---|---|---|
| baseline1_vector | 0.5903 | 0.6550 | 0.0048 |
| baseline2_llm | 0.0830 | 0.1050 | 0.0002 |
| full | 0.6932 | 0.7650 | 0.0048 |
| ablation_noprofile | 0.6527 | 0.7200 | 0.0056 |

## Per-category Precision@5 / NDCG@5

| Model | scene P@5 / NDCG | mood P@5 / NDCG | genre P@5 / NDCG | ambiguous P@5 / NDCG |
|---|---|---|---|---|
| baseline1_vector | 0.7200 / 0.5906 | 0.6000 / 0.4854 | 0.8800 / 0.8777 | 0.4200 / 0.4073 |
| baseline2_llm | 0.0800 / 0.0585 | 0.1400 / 0.1066 | 0.0600 / 0.0571 | 0.1400 / 0.1099 |
| full | 0.7800 / 0.6207 | 0.9200 / 0.8113 | 0.9400 / 0.9278 | 0.4200 / 0.4130 |
| ablation_noprofile | 0.8200 / 0.6499 | 0.6800 / 0.5730 | 0.9400 / 0.9425 | 0.4400 / 0.4455 |

## Baseline-2 hallucination rate (pure LLM, no RAG)

| Query | Unmatched (hallucinated) songs out of 5 |
|---|---|
| scene_01 | 4 |
| scene_02 | 3 |
| scene_03 | 5 |
| scene_04 | 5 |
| scene_05 | 3 |
| scene_06 | 4 |
| scene_07 | 5 |
| scene_08 | 5 |
| scene_09 | 4 |
| scene_10 | 4 |
| mood_01 | 3 |
| mood_02 | 5 |
| mood_03 | 4 |
| mood_04 | 3 |
| mood_05 | 4 |
| mood_06 | 2 |
| mood_07 | 4 |
| mood_08 | 5 |
| mood_09 | 5 |
| mood_10 | 3 |
| genre_01 | 5 |
| genre_02 | 5 |
| genre_03 | 4 |
| genre_04 | 4 |
| genre_05 | 5 |
| genre_06 | 4 |
| genre_07 | 5 |
| genre_08 | 5 |
| genre_09 | 5 |
| genre_10 | 4 |
| vague_01 | 3 |
| vague_02 | 4 |
| vague_03 | 3 |
| vague_04 | 4 |
| vague_05 | 5 |
| vague_06 | 3 |
| vague_07 | 5 |
| vague_08 | 5 |
| vague_09 | 2 |
| vague_10 | 3 |
