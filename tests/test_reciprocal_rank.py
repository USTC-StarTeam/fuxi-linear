import math

import pytest
import torch

from generative_recommenders.data.eval import EvalState, eval_metrics_v2_from_tensors
from generative_recommenders.indexing.candidate_index import CandidateIndex
from generative_recommenders.indexing.mips_top_k import MIPSBruteForceTopK
from generative_recommenders.modeling.sequential.features import SequentialFeatures


class FixedQueryModel:
    def get_item_embeddings(self, ids):
        return torch.ones_like(ids, dtype=torch.float32).unsqueeze(-1)

    def encode(self, *, past_payloads, **kwargs):
        return past_payloads["queries"]


def evaluate(item_count, targets, *, directions=None, history=None, ratings=None,
             threshold=4, microbatch=None):
    ids = torch.arange(1, item_count + 1).reshape(1, -1)
    embeddings = torch.arange(item_count, 0, -1, dtype=torch.float32).reshape(1, -1, 1)
    index = CandidateIndex(ids, embeddings)
    top_k = MIPSBruteForceTopK(embeddings, ids)
    state = EvalState(set(ids[0].tolist()), index, top_k)
    batch_size = len(targets)
    queries = torch.tensor(directions or [1.0] * batch_size, dtype=torch.float32).reshape(-1, 1)
    past_ids = torch.full((batch_size, 1), history or 0, dtype=torch.long)
    features = SequentialFeatures(torch.ones(batch_size, dtype=torch.long), past_ids, None,
                                  {"queries": queries})
    target_ids = torch.tensor(targets).reshape(-1, 1)
    target_ratings = None if ratings is None else torch.tensor(ratings).reshape(-1, 1)

    ranked_ids, _, _ = index.get_top_k_outputs(
        query_embeddings=queries, k=min(item_count, 2500), top_k_module=top_k,
        invalid_ids=past_ids if history is not None else None,
    )
    ranks = [
        next((rank for rank, candidate in enumerate(row, 1) if candidate == target), None)
        for row, target in zip(ranked_ids.tolist(), targets)
    ]
    result = eval_metrics_v2_from_tensors(
        state, FixedQueryModel(), features, target_ids, target_ratings=target_ratings,
        min_positive_rating=threshold, filter_invalid_ids=history is not None,
        user_max_batch_size=microbatch,
    )
    return ranks, result


def assert_reciprocal_ranks(scores, ranks):
    expected = torch.tensor([0.0 if rank is None else 1 / rank for rank in ranks])
    torch.testing.assert_close(scores, expected)
    assert all(score == 0.0 for score, rank in zip(scores.tolist(), ranks) if rank is None)
    assert torch.all((0 <= scores) & (scores <= 1))


@pytest.mark.parametrize("item_count,targets,expected_ranks", [
    (1, [1, 2], [1, None]),
    (3, [1, 2, 3], [1, 2, 3]),
    (3, [1, 2, 3, 4, 5], [1, 2, 3, None, None]),
    (2501, [1, 2, 2500, 2501, 2502], [1, 2, 2500, None, None]),
])
def test_rank_hits_and_misses(item_count, targets, expected_ranks):
    ranks, result = evaluate(item_count, targets)
    assert ranks == expected_ranks
    assert_reciprocal_ranks(result["mrr"], ranks)


@pytest.mark.parametrize("microbatch", [None, 1, 2, 4])
def test_microbatch_partition_preserves_user_order(microbatch):
    ranks, result = evaluate(3, [1, 1, 4, 2, 5], directions=[1, -1, 1, -1, 1],
                             microbatch=microbatch)
    assert ranks == [1, 3, None, 2, None]
    assert_reciprocal_ranks(result["mrr"], ranks)


@pytest.mark.parametrize("ratings,threshold", [([5, 3, 4, 2, 5], 4), ([5, 3, 4, 2, 5], 5),
                                               ([3, 2, 1, 3, 2], 4)])
def test_filtered_mrr(ratings, threshold):
    ranks, result = evaluate(3, [1, 4, 2, 5, 6], ratings=ratings, threshold=threshold)
    selected_ranks = [rank for rank, rating in zip(ranks, ratings) if rating >= threshold]
    scores = result[f"mrr_>={threshold}"]
    assert_reciprocal_ranks(scores, selected_ranks)
    torch.testing.assert_close(scores, result["mrr"][torch.tensor(ratings) >= threshold])


def test_target_excluded_by_candidate_history():
    ranks, result = evaluate(2501, [1, 2, 2501], history=1)
    assert ranks == [None, 1, 2500]
    assert_reciprocal_ranks(result["mrr"], ranks)
    torch.testing.assert_close(result["hr@1"], torch.tensor([False, True, False]))


def test_hit_rate_and_ndcg_preserve_all_existing_outputs():
    targets = [1, 10, 11, 50, 51, 100, 101, 200, 201, 500, 501, 1000, 1001, 2500, 2501]
    ratings = [5, 4, 3] * 5
    ranks, result = evaluate(2501, targets, ratings=ratings, threshold=5)
    expected = {}

    def dcg_at(rank, k):
        return 0.0 if rank is None or rank > k else 1 / math.log2(rank + 1)

    for k in (1, 10, 50, 100, 200):
        expected[f"ndcg@{k}"] = torch.tensor([dcg_at(rank, k) for rank in ranks])
    for k in (1, 10, 50, 100, 200, 500, 1000):
        expected[f"hr@{k}"] = torch.tensor([rank is not None and rank <= k for rank in ranks])
    expected["ndcg@10_>=4"] = expected["ndcg@10"][torch.tensor(ratings) >= 4]
    for k in (10, 50):
        expected[f"hr@{k}_>=5"] = expected[f"hr@{k}"][torch.tensor(ratings) >= 5]

    assert set(result) == {*expected, "mrr", "mrr_>=5"}
    for key, tensor in expected.items():
        torch.testing.assert_close(result[key], tensor)
