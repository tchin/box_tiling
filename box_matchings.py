import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from matching_3d_utils import *

grid_dim = [4,4,4]

H = nx.grid_graph(grid_dim)
#
# G = nx.DiGraph()
# G.add_nodes_from(H)
# G.add_edges_from(filter(lambda x: x[0] < x[1], H.edges()))


# list of edges in matching as tuples
# tiling_edges = filter(lambda x: sum(x[0]) % 2 == 0, nx.bipartite.maximum_matching(H).items())

## 4 x 4 x 4 box with twist -1
tiling_edges = [((0, 1, 3), (0, 0, 3)), ((0, 3, 3), (0, 2, 3)), ((1, 0, 3), (2, 0, 3)), ((2, 1, 3), (1, 1, 3)), ((1, 2, 3), (2, 2, 3)), ((1, 3, 2), (1, 3, 3)), ((2, 3, 3), (3, 3, 3)), ((3, 0, 3), (3, 0, 2)), ((3, 2, 3), (3, 1, 3)), ((0, 0, 2), (0, 0, 1)), ((0, 1, 1), (0, 1, 2)), ((0, 2, 2), (0, 2, 1)), ((0, 3, 1), (0, 3, 2)), ((1, 1, 2), (1, 0, 2)), ((1, 2, 1), (1, 2, 2)), ((2, 0, 2), (2, 0, 1)), ((2, 2, 2), (2, 1, 2)), ((2, 3, 1), (2, 3, 2)), ((3, 1, 2), (3, 1, 1)), ((3, 3, 2), (3, 2, 2)), ((1, 0, 1), (1, 0, 0)), ((2, 1, 1), (1, 1, 1)), ((1, 3, 0), (1, 3, 1)), ((2, 2, 0), (2, 2, 1)), ((3, 0, 1), (3, 0, 0)), ((3, 2, 1), (3, 3, 1)), ((0, 0, 0), (0, 1, 0)), ((0, 2, 0), (0, 3, 0)), ((1, 1, 0), (1, 2, 0)), ((2, 0, 0), (2, 1, 0)), ((3, 3, 0), (2, 3, 0)), ((3, 1, 0), (3, 2, 0))]
test_trit_vertices = [(1,2,2),(1,2,1),(1,1,1),(2,1,1),(2,2,2),(2,1,2)]
test_flip_vertices = [(1,2,3),(2,2,3),(1,1,3),(2,1,3)]

tiling = nx.DiGraph()
# tiling.add_nodes_from(H.nodes(True))
tiling.add_edges_from(tiling_edges)

plot_3d_matching(tiling)

flip_component = explore_flip_component(tiling)
nx.draw_networkx(flip_component)
print(flip_component.nodes())
