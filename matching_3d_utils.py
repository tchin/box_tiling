import numpy as np
import networkx as nx

import sys
sys.path.insert(0, '../graph_drawing')
from plotly_visualize import visualize_graph_3d

def is_adjacent(u,v):
    diff = np.array(u) - np.array(v)
    diff_ind = np.nonzero(diff)[0]
    return len(diff_ind) == 1 and abs(diff[diff_ind]) == 1

def plot_3d_matching(tiling, filename="outputs/3dmatch.html"):
    visualize_graph_3d(nx.to_undirected(tiling), tiling.nodes(), [], filename, title="3d", coords={v: v for v in tiling.nodes()})

def vertices_have_flip(vertices, tiling):
    """
    Given a (directed) tiling graph and four vertices, check if there is a flip move
    between the vertices.

    Assumes that the given tiling is indeed a perfect matching
    """
    assert len(vertices) == 4
    assert all([v in tiling for v in vertices])

    # check all edges out of a vertex in our group goes to another vertex in our group
    for v in vertices:
        neighbors = tiling.successors(v)
        if len(neighbors) > 0 and neighbors[0] not in vertices:
            return False

    return True

def execute_flip(vertices, tiling):
    assert vertices_have_flip(vertices, tiling)

    dif1 = (np.array(vertices[0])-np.array(vertices[1])) != 0
    dif2 = (np.array(vertices[0])-np.array(vertices[2])) != 0
    flip_dims = np.nonzero(np.logical_or(dif1,dif2))[0]

    edges_to_remove = []
    edges_to_add = []
    for v in vertices:
        # find vertex that v is currently connected to
        v2 = (list(tiling.predecessors(v)) + list(tiling.successors(v)))[0]

        # rotate v's edge to go in the other dimension
        cur_dim = np.nonzero(np.array(v)-np.array(v2))[0][0]
        new_dim = cur_dim ^ flip_dims[0] ^ flip_dims[1]

        # find vertex one over from v in new dimension
        new_end_list = list(v)
        new_end_list[new_dim] += 1
        new_end = tuple(new_end_list)

        # remove any outgoing edges from v
        edges_to_remove += [(v, u) for u in tiling.successors(v)]

        # if new_end is in our flip, add that edge. Otherwise, we'll add it
        # when we get to the vertex that will point to v
        if new_end in vertices:
            edges_to_add.append((v, new_end))

    tiling.remove_edges_from(edges_to_remove)
    tiling.add_edges_from(edges_to_add)

def vertices_have_trit(vertices, tiling):
    assert len(vertices) == 6
    assert all([v in tiling for v in vertices])

    trit_options = {v: filter(lambda x: is_adjacent(v,x), vertices) for v in vertices}
    print(trit_options)

    for v in vertices:
        neighbors = list(tiling.predecessors(v)) + list(tiling.successors(v))
        assert len(neighbors) == 1
        if neighbors[0] not in vertices:
            return False
    return True

def execute_trit(vertices, tiling):
    assert vertices_have_trit(vertices, tiling)
    trit_options = {v: filter(lambda x: is_adjacent(v,x), vertices) for v in vertices}

    edges_to_add = []
    edges_to_remove = []
    for v in vertices:
        edges_to_remove += [(v,u) for u in tiling.successors(v)]

        # guaranteed to be nonempty because we asserted this when checking a trit existed
        neighbors = list(tiling.predecessors(v)) + list(tiling.successors(v))

        new_end = tuple(np.bitwise_xor(np.bitwise_xor(neighbors[0],\
                                       trit_options[v][0]),\
                                       trit_options[v][1]))
        if v < new_end:
            edges_to_add.append((v, new_end))
    print("removing: " + str(edges_to_remove))
    print("adding: " + str(edges_to_add))
    tiling.remove_edges_from(edges_to_remove)
    tiling.add_edges_from(edges_to_add)

def compute_twist(tiling):
    u = np.array([0,0,-1])
    twist = 0

    for e in tiling.edges():
        edge_vect = np.array(e[1]) - np.array(e[0])

        #only need to consider non-vertical dominoes
        if not np.array_equal(np.abs(edge_vect), np.array([0,0,1])):
            shadow_dominoes = []
            for z in range(e[0][2]):
                v0 = e[0][:2] + (z,)
                v1 = e[1][:2] + (z,)
                shadow_dominoes += list(tiling.edges(v0)) + list(tiling.in_edges(v0))
                shadow_dominoes += list(tiling.edges(v1)) + list(tiling.in_edges(v1))
            shadow_vects = [np.array(d[1])-np.array(d[0]) for d in shadow_dominoes]
            contrib = 0
            print("edge: " + str(e))
            for d, v in zip(shadow_dominoes, shadow_vects):
                indiv_contrib = np.linalg.det(np.array([v, edge_vect, u]))
                contrib += indiv_contrib
                print("\t" + str(d) + " (" + str(v)+ "):\t" + str(indiv_contrib))
            print("\ttotal:" + str(contrib))
            twist += contrib
    return twist
