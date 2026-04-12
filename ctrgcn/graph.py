"""
graph.py — COCO 17-joint graph topology for CTR-GCN.

Produces a (3, V, V) normalized adjacency matrix using spatial partitioning:
  A[0]: self-connections
  A[1]: edges toward root (centripetal)
  A[2]: edges away from root (centrifugal)

Root node: index 0 (nose) — topmost node in COCO skeleton tree.
"""

import numpy as np


# COCO 17-joint skeleton edges (undirected, 0-indexed)
# Each tuple is (child, parent) in the skeleton tree rooted at nose (0)
INWARD_EDGES = [
    (1, 0), (2, 0),    # eyes ← nose
    (3, 1), (4, 2),    # ears ← eyes
    (5, 6),            # left_shoulder — right_shoulder (symmetric; treat as 5→6)
    (7, 5), (9, 7),    # left_elbow ← left_shoulder, left_wrist ← left_elbow
    (8, 6), (10, 8),   # right_elbow ← right_shoulder, right_wrist ← right_elbow
    (11, 5), (12, 6),  # left_hip ← left_shoulder, right_hip ← right_shoulder
    (11, 12),          # left_hip — right_hip
    (13, 11), (15, 13),# left_knee ← left_hip, left_ankle ← left_knee
    (14, 12), (16, 14),# right_knee ← right_hip, right_ankle ← right_knee
]


class Graph:
    """
    COCO 17-joint graph with 3-way spatial partitioning.

    Usage:
        graph = Graph()
        A = graph.get_adjacency()  # (3, 17, 17) float32 numpy array
    """

    def __init__(self, num_node: int = 17, center: int = 0):
        self.num_node = num_node
        self.center   = center
        self._A = None  # cached

    def get_adjacency(self) -> np.ndarray:
        if self._A is None:
            self._A = self._build()
        return self._A

    def _build(self) -> np.ndarray:
        V = self.num_node
        self_link    = [(i, i) for i in range(V)]
        outward      = [(j, i) for (i, j) in INWARD_EDGES]  # parent→child direction

        # Spatial partitioning: for each edge (i,j), determine "toward center" vs "away"
        # We BFS from center to assign distances
        dist = self._bfs_distance()

        A = np.zeros((3, V, V), dtype=np.float32)

        # Partition 0: self-connections
        for i, j in self_link:
            A[0, i, j] = 1.0

        # Partition 1: centripetal (child → parent, i.e., moving toward root)
        for i, j in INWARD_EDGES:
            # i = child (farther from root), j = parent (closer to root)
            # Edge direction i→j is centripetal
            A[1, i, j] = 1.0

        # Partition 2: centrifugal (parent → child, moving away from root)
        for i, j in outward:
            A[2, i, j] = 1.0

        # Normalize each partition: D^{-1/2} A D^{-1/2}
        for k in range(3):
            A[k] = _normalize(A[k])

        return A

    def _bfs_distance(self) -> dict:
        """BFS from center to compute graph distance for each node."""
        V = self.num_node
        adj = {i: set() for i in range(V)}
        for i, j in INWARD_EDGES:
            adj[i].add(j)
            adj[j].add(i)

        dist = {self.center: 0}
        queue = [self.center]
        while queue:
            node = queue.pop(0)
            for nb in adj[node]:
                if nb not in dist:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
        return dist


def _normalize(A: np.ndarray) -> np.ndarray:
    """D^{-1/2} A D^{-1/2} normalization."""
    rowsum = A.sum(axis=1)
    # Use errstate to suppress divide-by-zero for isolated nodes (set to 0 afterward)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_inv_sqrt = np.where(rowsum > 0, np.power(rowsum, -0.5, where=rowsum > 0), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt
