"""aggregate 单元测试：等价归簇与投票。"""
from jiushao.aggregate import cluster_and_vote


class TestClusterAndVote:
    def test_equiv_answers_merge(self):
        agg = cluster_and_vote(["1/2", "0.5", r"\frac{1}{2}", "2"])
        assert agg["final"] in ("1/2", "0.5", r"\frac{1}{2}")
        assert agg["clusters"][0][1] == 3

    def test_majority_wins(self):
        agg = cluster_and_vote(["7", "7", "8"])
        assert agg["final"] == "7"

    def test_none_filtered(self):
        agg = cluster_and_vote([None, "", "5"])
        assert agg["final"] == "5"
        assert agg["clusters"] == [("5", 1)]

    def test_all_none(self):
        agg = cluster_and_vote([None, None])
        assert agg["final"] is None and agg["clusters"] == []

    def test_clusters_sorted_by_votes(self):
        agg = cluster_and_vote(["a", "b", "b", "b", "a", "c"])
        votes = [n for _, n in agg["clusters"]]
        assert votes == sorted(votes, reverse=True) == [3, 2, 1]
