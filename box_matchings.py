import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from matching_3d_utils import *

grid_dim = [3,3,4]

H = nx.grid_graph(grid_dim)

G = nx.DiGraph()
G.add_nodes_from(H)
G.add_edges_from(filter(lambda x: x[0] < x[1], H.edges()))


# list of edges in matching as tuples
# tiling_edges = filter(lambda x: x[0] < x[1], nx.bipartite.maximum_matching(H).items())

## 3 x 3 x 2 box with twist 1
# tiling_edges = [((0,0,0), (1,0,0)), ((0,1,0),(0,2,0)), ((1,2,0),(2,2,0)), ((2,0,0),(2,1,0)),\
#                 ((0,0,1),(0,1,1)), ((0,2,1),(1,2,1)), ((2,1,1),(2,2,1)), ((1,0,1),(2,0,1)),\
#                 ((1,1,0),(1,1,1)),\
#                 ((0,0,2), (1,0,2)), ((0,1,2),(0,2,2)), ((1,2,2),(2,2,2)), ((2,0,2),(2,1,2)),\
#                 ((0,0,3),(0,1,3)), ((0,2,3),(1,2,3)), ((2,1,3),(2,2,3)), ((1,0,3),(2,0,3)),\
#                 ((1,1,2),(1,1,3))]

## 4 x 4 x 4 box with twist 1
tiling_edges = [((0,0,3),(0,1,3)), ((0,2,3),(0,3,3)), ((1,0,3),(2,0,3)), ((1,1,3),(2,1,3)), ((1,2,3),(2,2,3)), ((1,3,3),(1,3,2)), ((2,3,3),(3,3,3)), ((3,0,3),(3,0,2)), ((3,1,3), (3,2,3)),((0,0,2),(0,0,1)), ((0,1,2),(0,1,1)), ((0,2,2),(0,2,1)), ((0,3,2),(0,3,1)), ((1,0,2),(1,1,2)), ((1,2,2),(1,2,1)), ((2,0,2),(2,0,1)), ((2,1,2),(2,2,2)), ((2,3,2),(2,3,1)), ((3,1,2),(3,1,1)), ((3,2,2),(3,3,2)),((1,0,1),(1,0,0)), ((1,1,1),(2,1,1)), ((1,3,1),(1,3,0)), ((2,2,1),(2,2,0)), ((3,0,1),(3,0,0)), ((3,2,1),(3,3,1)),((0,0,0),(0,1,0)), ((0,2,0),(0,3,0)), ((1,1,0),(1,2,0)), ((2,0,0),(2,1,0)), ((2,3,0),(3,3,0)), ((3,1,0),(3,2,0))]
tiling = nx.DiGraph()
tiling.add_nodes_from(H.nodes(True))
tiling.add_edges_from(tiling_edges)

print(tiling.nodes(True))
print(tiling[(0,0,0)])

plot_3d_matching(tiling)
# test_flip_vertices = [(1,0,0),(0,0,0),(1,0,1),(0,0,1)]
# bottom_left_flip = vertices_have_flip(test_flip_vertices, tiling)
# execute_flip(test_flip_vertices, tiling)
# plot_3d_matching(tiling, "outputs/post_flip.html")
# test_trit_vertices = [(0,0,0), (1,0,0), (1,1,0), (0,0,1), (0,1,1), (1,1,1)]
twist1 = compute_twist(tiling)
print("pretwist: " + str(twist1))
# print(vertices_have_trit(test_trit_vertices, tiling))
# execute_trit(test_trit_vertices, tiling)
# twist2 = compute_twist(tiling)
# print("post: " + str(twist2))
# plot_3d_matching(tiling, "outputs/post_trit.html")
