from __future__ import annotations

from .schema import ReasoningCluster, ReasoningPath
from .text_utils import weighted_jaccard


def _average_similarity(path: ReasoningPath, members: list[ReasoningPath]) -> float:
    if not members:
        return 0.0
    return sum(weighted_jaccard(path.text, member.text) for member in members) / len(members)


def _select_medoid(members: list[ReasoningPath]) -> ReasoningPath:
    return max(
        members,
        key=lambda candidate: (_average_similarity(candidate, members), candidate.score_hint, candidate.path_id),
    )


def _merge_clusters(raw_clusters: list[list[ReasoningPath]], max_clusters: int) -> list[list[ReasoningPath]]:
    clusters = [list(cluster) for cluster in raw_clusters]
    while len(clusters) > max_clusters:
        best_pair: tuple[int, int] | None = None
        best_score = -1.0
        for left in range(len(clusters)):
            left_rep = _select_medoid(clusters[left])
            for right in range(left + 1, len(clusters)):
                right_rep = _select_medoid(clusters[right])
                score = weighted_jaccard(left_rep.text, right_rep.text)
                if score > best_score:
                    best_score = score
                    best_pair = (left, right)
        if best_pair is None:
            break
        left, right = best_pair
        clusters[left].extend(clusters[right])
        clusters.pop(right)
    return clusters


def cluster_reasoning_paths(
    paths: list[ReasoningPath],
    similarity_threshold: float,
    max_clusters: int,
) -> list[ReasoningCluster]:
    if not paths:
        return []

    raw_clusters: list[list[ReasoningPath]] = []
    for path in paths:
        best_cluster_index = -1
        best_similarity = -1.0
        for index, members in enumerate(raw_clusters):
            representative = _select_medoid(members)
            similarity = weighted_jaccard(path.text, representative.text)
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster_index = index
        if best_cluster_index >= 0 and best_similarity >= similarity_threshold:
            raw_clusters[best_cluster_index].append(path)
        else:
            raw_clusters.append([path])

    raw_clusters = _merge_clusters(raw_clusters, max_clusters=max_clusters)
    total_paths = float(len(paths))
    clusters: list[ReasoningCluster] = []
    for index, members in enumerate(raw_clusters):
        representative = _select_medoid(members)
        cohesion = _average_similarity(representative, members)
        clusters.append(
            ReasoningCluster(
                cluster_id=f"cluster_{index}",
                representative=representative,
                members=members,
                weight=len(members) / total_paths,
                cohesion=cohesion,
            )
        )

    return sorted(clusters, key=lambda cluster: (-cluster.weight, -cluster.cohesion, cluster.cluster_id))
