"""候选聚合：等价归簇 + 多数投票。"""
from .judge import equiv


def cluster_and_vote(answers: list[str | None]) -> dict:
    """把候选答案按数学等价归簇，返回 {final, clusters: [(代表答案, 票数)]}。"""
    clusters: list[list[str]] = []
    for ans in answers:
        if ans is None or not str(ans).strip():
            continue
        for c in clusters:
            if equiv(ans, c[0]):
                c.append(ans)
                break
        else:
            clusters.append([ans])
    if not clusters:
        return {"final": None, "clusters": []}
    clusters.sort(key=len, reverse=True)
    return {
        "final": clusters[0][0],
        "clusters": [(c[0], len(c)) for c in clusters],
    }
