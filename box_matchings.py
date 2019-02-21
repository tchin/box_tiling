import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from matching_3d_utils import *

grid_dim = [2,2,2]

H = nx.grid_graph(grid_dim)
#
# G = nx.DiGraph()
# G.add_nodes_from(H)
# G.add_edges_from(filter(lambda x: x[0] < x[1], H.edges()))


# list of edges in matching as tuples
# tiling_edges = filter(lambda x: sum(x[0]) % 2 == 0, nx.bipartite.maximum_matching(H).items())

## 4 x 4 x 4 box with twist -1
# tiling_edges = [((0, 1, 3), (0, 0, 3)), ((0, 3, 3), (0, 2, 3)), ((1, 0, 3), (2, 0, 3)), ((2, 1, 3), (1, 1, 3)), ((1, 2, 3), (2, 2, 3)), ((1, 3, 2), (1, 3, 3)), ((2, 3, 3), (3, 3, 3)), ((3, 0, 3), (3, 0, 2)), ((3, 2, 3), (3, 1, 3)), ((0, 0, 2), (0, 0, 1)), ((0, 1, 1), (0, 1, 2)), ((0, 2, 2), (0, 2, 1)), ((0, 3, 1), (0, 3, 2)), ((1, 1, 2), (1, 0, 2)), ((1, 2, 1), (1, 2, 2)), ((2, 0, 2), (2, 0, 1)), ((2, 2, 2), (2, 1, 2)), ((2, 3, 1), (2, 3, 2)), ((3, 1, 2), (3, 1, 1)), ((3, 3, 2), (3, 2, 2)), ((1, 0, 1), (1, 0, 0)), ((2, 1, 1), (1, 1, 1)), ((1, 3, 0), (1, 3, 1)), ((2, 2, 0), (2, 2, 1)), ((3, 0, 1), (3, 0, 0)), ((3, 2, 1), (3, 3, 1)), ((0, 0, 0), (0, 1, 0)), ((0, 2, 0), (0, 3, 0)), ((1, 1, 0), (1, 2, 0)), ((2, 0, 0), (2, 1, 0)), ((3, 3, 0), (2, 3, 0)), ((3, 1, 0), (3, 2, 0))]
# test_trit_vertices = [(1,2,2),(1,2,1),(1,1,1),(2,1,1),(2,2,2),(2,1,2)]
# test_flip_vertices = [(1,2,3),(2,2,3),(1,1,3),(2,1,3)]

## 4x4x4 box with twist 1. 236 tilings in flip component
# tiling_edges = [\
#     ((0,0,0),(1,0,0)), ((2,0,0),(2,1,0)), ((1,1,0),(1,2,0)), ((3,1,0),(3,2,0)), ((0,2,0),(0,3,0)), ((3,3,0),(2,3,0)),\
#     ((0,1,1),(0,1,0)), ((1,3,0),(1,3,1)), ((2,2,0),(2,2,1)), ((3,0,1),(3,0,0)),\
#     ((1,0,1),(1,1,1)), ((2,1,1),(3,1,1)), ((3,2,1),(3,3,1)),\
#     ((0,0,2),(0,0,1)), ((2,0,2),(2,0,1)), ((0,2,2),(0,2,1)), ((1,2,1),(1,2,2)), ((0,3,1),(0,3,2)), ((2,3,1),(2,3,2)),\
#     ((1,1,2),(0,1,2)), ((2,2,2),(3,2,2)), ((3,1,2),(3,0,2)),\
#     ((1,0,3),(1,0,2)), ((1,3,2),(1,3,3)), ((2,1,3),(2,1,2)), ((3,3,2),(3,3,3)),\
#     ((0,1,3),(0,0,3)), ((0,3,3),(0,2,3)), ((1,2,3),(1,1,3)), ((2,3,3),(2,2,3)), ((3,0,3),(2,0,3)), ((3,2,3),(3,1,3))\
#     ]

## 4x4x4 box with twist 0 in giant (4412646453) component. Crashes if you try to explore the full component
tiling_edges = orient_edges([((x,y,z), (x,y,z+1)) for x in range(4) for y in range(4) for z in [0,2]])

assert(nx.matching.is_perfect_matching(H, set(tiling_edges)))
assert(is_oriented(tiling_edges))

tiling = nx.DiGraph()
# tiling.add_nodes_from(H.nodes(True))
tiling.add_edges_from(tiling_edges)

#plot_3d_matching(tiling)
print("starting exploration")
flip_component = explore_flip_component(tiling, progress = 10000)
print(nx.number_of_nodes(flip_component))
