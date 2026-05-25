"""Probabilistic Roadmap (PRM) joint-space planner.

Builds a reusable roadmap of collision-free configurations once, then answers
multiple start/goal queries by connecting them to the roadmap and running a
shortest-path search. Uses the shared CollisionChecker for node/edge validity.
"""

import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors

from .collision import CollisionChecker


class PRMPlanner:
    def __init__(self, model, data, kinematics, num_samples=400, k_neighbors=10, seed=0):
        self.model = model
        self.data = data
        self.kin = kinematics
        self.num_samples = num_samples
        self.k = k_neighbors
        self.checker = CollisionChecker(model, data, kinematics)
        self.lo = kinematics.jnt_low
        self.hi = kinematics.jnt_high
        self.rng = np.random.default_rng(seed)
        self.nodes = []
        self.graph = nx.Graph()

    def build_roadmap(self):
        print(f"PRM: sampling {self.num_samples} collision-free nodes...")
        self.nodes = []
        self.graph.clear()
        while len(self.nodes) < self.num_samples:
            q = self.rng.uniform(self.lo, self.hi)
            if not self.checker.in_collision(q):
                self.graph.add_node(len(self.nodes))
                self.nodes.append(q)
        nbrs = NearestNeighbors(n_neighbors=min(self.k, len(self.nodes))).fit(self.nodes)
        dist, idx = nbrs.kneighbors(self.nodes)
        for i in range(len(self.nodes)):
            for jj in range(1, idx.shape[1]):
                j = int(idx[i][jj])
                if not self.graph.has_edge(i, j) and self.checker.edge_clear(self.nodes[i], self.nodes[j]):
                    self.graph.add_edge(i, j, weight=float(dist[i][jj]))
        print(f"PRM: roadmap has {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges.")

    def plan(self, q_start, q_goal):
        if not self.nodes:
            self.build_roadmap()
        if self.checker.in_collision(q_start) or self.checker.in_collision(q_goal):
            print("PRM: start or goal in collision."); return None

        nodes = self.nodes + [np.asarray(q_start, float), np.asarray(q_goal, float)]
        s, g = len(nodes) - 2, len(nodes) - 1
        self.graph.add_node(s); self.graph.add_node(g)
        nbrs = NearestNeighbors(n_neighbors=min(self.k, len(self.nodes))).fit(self.nodes)
        for idx_q, q in ((s, nodes[s]), (g, nodes[g])):
            dist, nb = nbrs.kneighbors([q])
            for kk in range(nb.shape[1]):
                j = int(nb[0][kk])
                if self.checker.edge_clear(q, self.nodes[j]):
                    self.graph.add_edge(idx_q, j, weight=float(dist[0][kk]))
        try:
            ids = nx.shortest_path(self.graph, s, g, weight="weight")
            path = [nodes[i] for i in ids]
        except nx.NetworkXNoPath:
            print("PRM: no path."); path = None
        self.graph.remove_node(s); self.graph.remove_node(g)
        return path
