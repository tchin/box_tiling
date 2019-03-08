import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from matching_3d_utils import *


def color_matching(edge_tuple, color_trits=False):
    edge_dim = get_edge_dim(edge_tuple[0][0], edge_tuple[0][1])
    for e in edge_tuple:
        if e[0][edge_dim] == e[1][edge_dim]:
            trits = get_all_trits_edges(edge_tuple)
            print(trits)
            if color_trits and trits != []:
                print("trit found: " + str(edge_tuple))
                return (1.0, 1.0, 0.0)
            else:
                return (0.5, 0.5, 0.5)
    c = [0.0,0.0,0.0]
    c[edge_dim] += 1.0
    return tuple(c)


## 2 x 2 x 2, all parallel
# H = nx.grid_graph([2,2,2])
# tiling_edges = orient_edges([((x,y,0), (x,y,1)) for x in [0,1] for y in [0,1]])

## 2 x 2 x 3, all parallel
# H = nx.grid_graph([3,2,2]) # networkx reverses the order of the dimensions
# tiling_edges = orient_edges([((0,y,z), (1,y,z)) for y in range(2) for z in range(3)])

## 3 x 3 x 2, all parallel
# H = nx.grid_graph([2,3,3])
# tiling_edges = orient_edges([((x,y,0), (x,y,1)) for x in range(3) for y in range(3)])

## 4 x 4 x 4 box with twist -1
# H = nx.grid_graph([4,4,4])
# tiling_edges = [((0, 1, 3), (0, 0, 3)), ((0, 3, 3), (0, 2, 3)), ((1, 0, 3), (2, 0, 3)), ((2, 1, 3), (1, 1, 3)), ((1, 2, 3), (2, 2, 3)), ((1, 3, 2), (1, 3, 3)), ((2, 3, 3), (3, 3, 3)), ((3, 0, 3), (3, 0, 2)), ((3, 2, 3), (3, 1, 3)), ((0, 0, 2), (0, 0, 1)), ((0, 1, 1), (0, 1, 2)), ((0, 2, 2), (0, 2, 1)), ((0, 3, 1), (0, 3, 2)), ((1, 1, 2), (1, 0, 2)), ((1, 2, 1), (1, 2, 2)), ((2, 0, 2), (2, 0, 1)), ((2, 2, 2), (2, 1, 2)), ((2, 3, 1), (2, 3, 2)), ((3, 1, 2), (3, 1, 1)), ((3, 3, 2), (3, 2, 2)), ((1, 0, 1), (1, 0, 0)), ((2, 1, 1), (1, 1, 1)), ((1, 3, 0), (1, 3, 1)), ((2, 2, 0), (2, 2, 1)), ((3, 0, 1), (3, 0, 0)), ((3, 2, 1), (3, 3, 1)), ((0, 0, 0), (0, 1, 0)), ((0, 2, 0), (0, 3, 0)), ((1, 1, 0), (1, 2, 0)), ((2, 0, 0), (2, 1, 0)), ((3, 3, 0), (2, 3, 0)), ((3, 1, 0), (3, 2, 0))]
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

# print("starting exploration")
flip_comp = explore_flip_component(tiling, progress=10000)
nodes = flip_comp.nodes()
colors = [color_matching(t, color_trits=True) for t in nodes]
sizes = [1 if c == (.5, .5, .5) else 100 for c in colors]
plt.figure()
nx.draw(flip_comp, node_list=nodes, node_size=sizes, node_color=colors, with_labels=False)
plt.title("3 x 2 x 2 tilings")
plt.show(block=False)

nx.write_adjlist(flip_comp, "flip_components/graphs/box322comp1.txt")
plt.savefig("flip_components/pictures/box322comp1.png")
