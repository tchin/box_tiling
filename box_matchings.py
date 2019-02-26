import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

from matching_3d_utils import *

## 4 x 4 x 4 box with twist -1
H = nx.grid_graph([4,4,4])
tiling_edges = [((0, 1, 3), (0, 0, 3)), ((0, 3, 3), (0, 2, 3)), ((1, 0, 3), (2, 0, 3)), ((2, 1, 3), (1, 1, 3)), ((1, 2, 3), (2, 2, 3)), ((1, 3, 2), (1, 3, 3)), ((2, 3, 3), (3, 3, 3)), ((3, 0, 3), (3, 0, 2)), ((3, 2, 3), (3, 1, 3)), ((0, 0, 2), (0, 0, 1)), ((0, 1, 1), (0, 1, 2)), ((0, 2, 2), (0, 2, 1)), ((0, 3, 1), (0, 3, 2)), ((1, 1, 2), (1, 0, 2)), ((1, 2, 1), (1, 2, 2)), ((2, 0, 2), (2, 0, 1)), ((2, 2, 2), (2, 1, 2)), ((2, 3, 1), (2, 3, 2)), ((3, 1, 2), (3, 1, 1)), ((3, 3, 2), (3, 2, 2)), ((1, 0, 1), (1, 0, 0)), ((2, 1, 1), (1, 1, 1)), ((1, 3, 0), (1, 3, 1)), ((2, 2, 0), (2, 2, 1)), ((3, 0, 1), (3, 0, 0)), ((3, 2, 1), (3, 3, 1)), ((0, 0, 0), (0, 1, 0)), ((0, 2, 0), (0, 3, 0)), ((1, 1, 0), (1, 2, 0)), ((2, 0, 0), (2, 1, 0)), ((3, 3, 0), (2, 3, 0)), ((3, 1, 0), (3, 2, 0))]
# test_trit_vertices = [(1,2,2),(1,2,1),(1,1,1),(2,1,1),(2,2,2),(2,1,2)]
# test_flip_vertices = [(1,2,3),(2,2,3),(1,1,3),(2,1,3)]

## 4x4x4 box with twist 1. 236 tilings in flip component
# H = nx.grid_graph([4,4,4])
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
# H = nx.grid_graph([4,4,4])
# tiling_edges = orient_edges([((x,y,z), (x,y,z+1)) for x in range(4) for y in range(4) for z in [0,2]])

assert(nx.matching.is_perfect_matching(H, set(tiling_edges)))
assert(is_oriented(tiling_edges))

tiling = nx.DiGraph()
tiling.add_edges_from(tiling_edges)

trits = get_all_trits_edges(tiling_edges)
for t in trits:
    print(sorted(t))
plot_3d_matching(tiling)
# print("starting exploration")
# flip_set = get_flip_component(tiling, progress=10000)
# print("component size: " + str(len(flip_set)))
