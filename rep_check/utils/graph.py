import numpy as np
from typing import Optional


class MediaPipeGraph():
    
    def __init__(
        self,
        num_node: Optional[int] = 23,
        max_hop: Optional[int] = 1,
        dilation: Optional[int] = 1,
        strategy: Optional[str] = 'uniform'
    ):
        self.num_node = num_node
        self.max_hop = max_hop
        self.dilation = dilation
        self.strategy = strategy

    @property
    def edge(self):
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [
            # Upper body
            (0, 2), (2, 4),                           # Left shoulder → elbow → wrist
            (1, 3), (3, 5),                           # Right shoulder → elbow → wrist
            (4, 6), (4, 8), (4, 10),                  # Left wrist → pinky/index/thumb
            (5, 7), (5, 9), (5, 11),                  # Right wrist → pinky/index/thumb
            (0, 1),                                   # Shoulders connection
            # Torso and pelvis
            (0, 12), (1, 13),                         # Left/Right shoulders → hips
            (12, 22), (13, 22),                       # Left/Right hip → pelvis
            # Lower body
            (12, 14), (14, 16), (16, 18), (16, 20),   # Left hip → knee → ankle → heel/foot
            (13, 15), (15, 17), (17, 19), (17, 21),   # Right hip → knee → ankle → heel/foot
        ]
        return self_link + neighbor_link
    
    @property
    def hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        # Compute hop steps
        hop_dist = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dist[arrive_mat[d]] = d
        return hop_dist

    @property
    def A(self):  # Adjacency matrix
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_distance == hop] = 1
        normalized_adjacency = normalize_digraph(adjacency)
        if self.strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalized_adjacency
        elif self.strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dist == hop] = normalized_adjacency[self.hop_dist == hop]
        elif self.strategy == 'spatial':
            center_id = 22
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_distance[j, i] == hop:
                            if self.hop_distance[j, center_id] == self.hop_distance[i, center_id]:
                                a_root[j, i] = normalized_adjacency[j, i]
                            elif self.hop_distance[j, center_id] > self.hop_distance[i, center_id]:
                                a_close[j, i] = normalized_adjacency[j, i]
                            else:
                                a_further[j, i] = normalized_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
        return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD