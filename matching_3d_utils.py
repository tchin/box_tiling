import numpy as np
import networkx as nx
import itertools

import sys
sys.path.insert(0, '../graph_drawing')
from plotly_visualize import visualize_graph_3d

def is_adjacent(u,v):
    diff = np.array(u) - np.array(v)
    diff_ind = np.nonzero(diff)[0]
    return len(diff_ind) == 1 and abs(diff[diff_ind]) == 1

def get_edge_dim(u,v):
    assert(is_adjacent(u,v))
    diff = np.array(u) - np.array(v)
    diff_ind = np.nonzero(diff)[0]
    return diff_ind[0]

def orient_edges(tiling):
    oriented = tiling
    for i in range(len(tiling)):
        if sum(tiling[i][0]) % 2 == 1:
            assert(sum(tiling[i][1]) % 2 == 0)
            e = (tiling[i][1], tiling[i][0])
            oriented[i] = e
    return oriented

def is_oriented_digraph(tiling):
    return is_oriented(tiling.edges())

def is_oriented(tiling):
    return all([sum(e[0]) % 2 == 0 for e in tiling])

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
        neighbors = list(tiling.successors(v)) + list(tiling.predecessors(v))
        assert(len(neighbors) == 1)
        if neighbors[0] not in vertices:
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
            if sum(v) % 2 == 0:
                edges_to_add.append((v, new_end))
            else:
                edges_to_add.append((new_end,v))

    tiling.remove_edges_from(edges_to_remove)
    tiling.add_edges_from(edges_to_add)

def execute_flip_edges(vertices, tiling_edges):
    dif1 = (np.array(vertices[0])-np.array(vertices[1])) != 0
    dif2 = (np.array(vertices[0])-np.array(vertices[2])) != 0
    flip_dims = np.nonzero(np.logical_or(dif1,dif2))[0]

    sources = filter(lambda x: sum(x) % 2 == 0, vertices)

    new_edges = []
    for e in tiling_edges:
        if e[0] in sources:
            cur_dim = get_edge_dim(e[0],e[1])
            new_dim = cur_dim ^ flip_dims[0] ^ flip_dims[1]

            new_end_list = list(e[0])
            new_end_list[new_dim] += 1
            if tuple(new_end_list) in vertices:
                new_edges.append((e[0], tuple(new_end_list)))
            else:
                new_end_list[new_dim] -= 2
                assert(tuple(new_end_list) in vertices)
                new_edges.append((e[0], tuple(new_end_list)))
        else:
            new_edges.append(e)
    return sorted(new_edges)


def vertices_have_trit(vertices, tiling):
    assert len(vertices) == 6
    assert all([v in tiling for v in vertices])

    trit_options = {v: filter(lambda x: is_adjacent(v,x), vertices) for v in vertices}

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
        if sum(v) % 2 == 0:
            edges_to_add.append((v, new_end))
    tiling.remove_edges_from(edges_to_remove)
    tiling.add_edges_from(edges_to_add)

def get_all_flips(tiling):
    flips = []

    # index flips by vertex with lowest coordinates
    for v in tiling.nodes():
        v_arr = np.array(v)
        x_neighbor = tuple(v_arr + np.array([1,0,0]))
        y_neighbor = tuple(v_arr + np.array([0,1,0]))
        z_neighbor = tuple(v_arr + np.array([0,0,1]))

        xy = tuple(v_arr + np.array([1,1,0]))
        yz = tuple(v_arr + np.array([0,1,1]))
        xz = tuple(v_arr + np.array([1,0,1]))

        if x_neighbor in tiling and y_neighbor in tiling:
            f = [v, x_neighbor, y_neighbor, xy]
            if vertices_have_flip(f, tiling):
                flips.append(f)
        if x_neighbor in tiling and z_neighbor in tiling:
            f = [v, x_neighbor, z_neighbor, xz]
            if vertices_have_flip(f, tiling):
                flips.append(f)
        if y_neighbor in tiling and z_neighbor in tiling:
            f = [v, y_neighbor, z_neighbor, yz]
            if vertices_have_flip(f, tiling):
                flips.append(f)
    return flips

def get_all_flips_edges(tiling_edges):
    source_to_head = dict(tiling_edges)
    head_to_source = dict((h,s) for s,h in tiling_edges)

    flips = []
    for source in source_to_head:
        cur_dim = get_edge_dim(source, source_to_head[source])

        neighbors_2d = [[0,1],[1,0]]
        for n in neighbors_2d:
            n.insert(cur_dim,0)
        neighbors_to_check = [tuple(np.array(source) + np.array(n)) for n in neighbors_2d]

        for n in neighbors_to_check:
            if n in head_to_source:
                n_edge = (head_to_source[n], n)
                if is_adjacent(n_edge[0], source_to_head[source]):
                    flips.append([source, source_to_head[source], n, head_to_source[n]])
    return flips


def explore_flip_component(tiling, progress=None):
    G = nx.Graph()
    G.add_node(tuple(sorted(tiling.edges())))

    q = [tiling]
    count = 0
    while q: #continue until queue is empty
        cur = q[0]
        cur_node = tuple(sorted(cur.edges()))
        flips = get_all_flips(cur)

        done = [G[cur_node][v]['flip'] for v in G[cur_node]]
        flips = filter(lambda x: sorted(x) not in done, flips)

        for flip in flips:
            flip = sorted(flip)
            new_tiling = nx.DiGraph()
            new_tiling.add_nodes_from(cur)
            new_tiling.add_edges_from(cur.edges())
            execute_flip(flip, new_tiling)
            new_node = tuple(sorted(new_tiling.edges()))

            if new_node not in G:
                G.add_node(new_node)
                q.append(new_tiling)
                if progress != None:
                    count += 1
                    if count % progress == 0:
                        print(count)
            G.add_edge(cur_node, new_node, flip = flip)
        q = q[1:]
    return G

def get_flip_component(tiling, progress=None):
    component = {tuple(sorted(tiling.edges())): []}
    count = 1

    q = [tuple(sorted(tiling.edges()))]
    while q:
        cur = q[0]

        flips = filter(lambda x: x not in component[cur], get_all_flips_edges(cur))
        for flip in flips:
            flip = sorted(flip)
            new_node = tuple(execute_flip_edges(flip, cur))
            if new_node not in component:
                component[new_node] = []
                q.append(new_node)
                if progress != None:
                    count += 1
                    if count % progress == 0:
                        print(count)
            component[new_node].append(flip)
        q = q[1:]
    return component

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
            for d, v in zip(shadow_dominoes, shadow_vects):
                twist += np.linalg.det(np.array([v, edge_vect, u]))
    return int(twist/4.0)
