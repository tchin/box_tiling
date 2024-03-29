import numpy as np
import networkx as nx
from collections import deque
import sqlite3
import sys


# return whether or not two nodes in the box are adjacent, either using provided dimensions, or by comparing np arrays
def is_adjacent(u,v, dims=None):
    if dims:
        n = dims[0] * dims[1] * dims[2]
        if u >= n or v >=n:
            return False
        poss = [1, dims[0], dims[0]*dims[1]]
        diff = abs(u-v)
        if diff not in poss:
            return False
        else:
            for i in range(len(poss)-1):
                if diff == poss[i]:
                    return int(u/poss[i+1]) == int(v/poss[i+1])
        return True
    else:
        diff = np.array(u) - np.array(v)
        diff_ind = np.nonzero(diff)[0]
        return len(diff_ind) == 1 and abs(diff[diff_ind]) == 1


# return differences that can correspond to offsets between neighboring vertices based on dimensions
def get_neighbor_offsets(dims):
    return [1, dims[0], dims[0]*dims[1]]


# determine along which dimension u and v are adjacent
def get_edge_dim(u,v):
    assert(is_adjacent(u,v))
    diff = np.array(u) - np.array(v)
    diff_ind = np.nonzero(diff)[0]
    return diff_ind[0]

# given vertex and an np array, return the vertex dims away from v
def increment_dims(v, dims):
    new_v = list(v)
    for d in dims:
        new_v[d] += 1
    return tuple(new_v)


# turn an unoriented tiling into an oriented one (to compute twist)
def orient_edges(tiling):
    oriented = tiling
    for i in range(len(tiling)):
        if sum(tiling[i][0]) % 2 == 1:
            assert(sum(tiling[i][1]) % 2 == 0)
            e = (tiling[i][1], tiling[i][0])
            oriented[i] = e
    return oriented


# convert a perfect matching graph into a tiling tuple
def graph_to_matching_tuple(tiling):
    return tuple([list(tiling.neighbors(v))[0] for v in range(nx.number_of_nodes(tiling))])


def is_oriented_digraph(tiling):
    return is_oriented(tiling.edges())


def is_oriented(tiling):
    return all([sum(e[0]) % 2 == 0 for e in tiling])


# check whether four vertices are tiled such that a flip move could be performed in a given tiling
def vertices_have_flip(vertices, tiling, n=None):
    """
    Given a (directed) tiling graph and four vertices, check if there is a flip move
    between the vertices.

    Assumes that the given tiling is indeed a perfect matching
    """
    if not n:
        n = len(tiling)
    # check all edges out of a vertex in our group goes to another vertex in our group
    for v in vertices:
        if v >= n:
            return False
        neighbor = tiling[v]
        if neighbor not in vertices:
            return False
    return True


# given four vertices with a flip move, return a new tiling resulting from executing a flip on those vertices
## uses np arrays for vertices and directed graph for tiling
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

# same as above, but using more efficient tuple storage
def execute_flip_edges(vertices, tiling_edges, dims):
    poss = [1, dims[0], dims[0] * dims[1]]
    n1 = tiling_edges[vertices[0]]

    new_neighbor = next(v for v in vertices if (v != n1 and abs(v-vertices[0]) in poss))
    non_neighbor = tiling_edges[new_neighbor]

    new_tiling = list(tiling_edges)
    new_tiling[vertices[0]] = new_neighbor
    new_tiling[new_neighbor] = vertices[0]
    new_tiling[n1] = non_neighbor
    new_tiling[non_neighbor] = n1

    return tuple(new_tiling)

# determine whether six vertices have a potential trit move
def vertices_have_trit(vertices, tiling):
    assert len(vertices) == 6
    assert all([v in tiling for v in vertices])

    trit_options = {v: filter(lambda x: is_adjacent(v,x), vertices) for v in vertices}

    for v in vertices:
        neighbors = list(tiling.predecessors(v)) + list(tiling.successors(v))
        assert(len(neighbors) == 1)
        if neighbors[0] not in vertices:
            return False
    return True

# execute a trit move on the given vertices
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


# return a list of lists, where each inner list is a possible flip move in the tiling
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

# same as above, but using more efficient tuple data structure
def get_all_flips_edges(tiling_edges, dims):
    flips = []
    n = len(tiling_edges)

    poss = get_neighbor_offsets(dims)
    for v in range(len(tiling_edges)):
        neighbor = tiling_edges[v]
        if neighbor > v:
            diff = neighbor - v
            for offset in poss:
                if (offset != diff and v + offset < n
                        and tiling_edges[v + offset] == neighbor + offset
                        and is_adjacent(v, v + offset, dims)):
                    flips.append([v, neighbor, v + offset, neighbor + offset])
    return flips


# return all possible trits, using tuple data structure
def get_all_trits_edges(tiling_edges):
    trits = []

    source_to_head = dict(tiling_edges)
    head_to_source = dict((h,s) for s,h in tiling_edges)

    for u in source_to_head:
        v = source_to_head[u]
        dim1 = get_edge_dim(u,v)
        other_dims = [0,1,2]
        other_dims.remove(dim1)
        dim2 = other_dims[0]
        dim3 = other_dims[1]

        u2 = increment_dims(u,[dim2])
        u3 = increment_dims(u,[dim3])
        u23 = increment_dims(u,[dim2,dim3])

        v2 = increment_dims(v,[dim2])
        v3 = increment_dims(v,[dim3])
        v23 = increment_dims(v, [dim2, dim3])
        if u23 in source_to_head and v23 in head_to_source:
            if source_to_head[u23] == u2 and source_to_head[v3] == v23:
                trits.append([u,v,u23,u2,v23,v3])
            elif source_to_head[u23] == u3 and source_to_head[v2] == v23:
                trits.append([u,v,u23,u3,v23,v2])
    return trits


# (ideally) return full graph of the flip connected component reachable from the initial tiling
## warning: hangs on large components due to lack of memory
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


# populate a database with all tilings flip-connected to initial tiling.
## warning: hangs on large inputs (queue is too large to fit in memory)
def get_flip_component_disk(tiling, dims, db, progress=None):
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS component(tiling text, flips text, depth integer)")
    cur.execute("CREATE INDEX IF NOT EXISTS tiling_index ON component(tiling)")

    cur.execute("SELECT * FROM component")
    start_row = cur.fetchone()
    if start_row:
        q = init_queue_from_disk(db)
        cur.execute("select max(depth) from component")
        cur_depth = cur.fetchone()[0]

    else:
        cur.execute("INSERT INTO component VALUES (?,?,?)", (str(tiling), "", 0))
        q = deque([tiling])
        cur_depth = 0
        count = 1

    f = open("")
    while q:
        cur_tiling = q.pop()

        cur.execute("SELECT * FROM component WHERE tiling = :tiling", {"tiling": str(cur_tiling)})
        row = cur.fetchone()

        if row["flips"] == "":
            done = []
        else:
            done = [sorted([int(i) for i in flip.split(",")]) for flip in row["flips"].split(":")]

        if row["depth"] > cur_depth:
            cur.execute("select count(*) from component where depth = ?", (cur_depth-1,))
            num_deleting = cur.fetchone()[0]
            print(str(num_deleting) + " tilings at depth " + str(cur_depth - 1))
            cur.execute("DELETE FROM component WHERE depth <= ?", (cur_depth-1,))
            cur_depth += 1

        flips = filter(lambda x: sorted(x) not in done, get_all_flips_edges(cur_tiling, dims))
        for flip in flips:
            flip = sorted(flip)
            flip_str = ",".join([str(f) for f in flip])

            new_node = execute_flip_edges(flip, cur_tiling, dims)

            cur.execute("SELECT * FROM component WHERE tiling=:tiling", {"tiling": str(new_node)})
            new_row = cur.fetchone()
            if not new_row:
                cur.execute("INSERT INTO component VALUES (?,?,?)", (str(new_node), flip_str, cur_depth+1))
                q.append(new_node)
                if progress:
                    count += 1
                    if count % progress == 0:
                        conn.commit()
                        print(count)
            else:
                new_flips = new_row["flips"] + ":" + flip_str
                cur.execute("UPDATE component SET flips = ? WHERE tiling = ?", (new_flips, str(cur_tiling)))

    print("final count: " + str(count))


# utility for restarting if get_flip_component_disk is interrupted
def init_queue_from_disk(db):
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("select max(depth) from component")
    max_depth = cur.fetchone()[0]

    cur.execute("select tiling from component where depth=?", (max_depth,))
    max_depth_tilings = cur.fetchall()
    tiling_list = [r["tiling"] for r in max_depth_tilings]
    qstr_list = [s[1:-1].split(",") for s in tiling_list]
    q_list = [tuple([int(i) for i in s]) for s in qstr_list]
    conn.close()

    return deque(q_list)


# count how many tilings are in the flip-connected component
def get_flip_component_size(tiling, dims, progress_file, progress=None):
    # clear file contents
    f = open(progress_file, "w+")
    f.close()

    count = 1
    prev_depth = {}
    cur_depth = {tiling: []}
    next_depth = {}
    depth = 0

    degree_seq = {}

    q = deque([tiling])
    progress_str = ""
    while q:
        cur = q.popleft()

        if cur in next_depth:
            progress_str += str(depth) + "," + str(len(cur_depth))+"\n"
            prev_depth.clear()
            prev_depth = cur_depth
            cur_depth = next_depth
            next_depth = dict()
            depth += 1

        all_flips = get_all_flips_edges(cur, dims)
        if len(all_flips) not in degree_seq:
            degree_seq[len(all_flips)] = 1
        else:
            degree_seq[len(all_flips)] += 1

        flips = filter(lambda x: sorted(x) not in cur_depth[cur], all_flips)
        for flip in flips:
            flip = sorted(flip)
            new_node = tuple(execute_flip_edges(flip, cur, dims))
            if new_node not in prev_depth and new_node not in cur_depth:
                if new_node not in next_depth:
                    next_depth[new_node] = []
                    q.append(new_node)
                    count += 1
                    if progress and count % progress == 0:
                        print(count)
                        f = open(progress_file, 'a+')
                        f.write(progress_str)
                        f.close()
                        progress_str = ""
                        sys.stdout.flush()
                next_depth[new_node].append(flip)
    f = open(progress_file, 'a+', buffering=1024)
    f.write(progress_str)
    f.close()
    print(degree_seq)
    return count


# get a list (set) of all tilings in the connected component
def get_flip_component(tiling, dims, progress=None):
    component = {tiling: []}
    count = 1

    q = deque([tiling])
    while q:
        cur = q.pop()

        flips = filter(lambda x: sorted(x) not in component[cur], get_all_flips_edges(cur, dims))
        for flip in flips:
            flip = sorted(flip)
            new_node = tuple(execute_flip_edges(flip, cur, dims))
            if new_node not in component:
                component[new_node] = []
                q.append(new_node)
                if progress:
                    count += 1
                    if count % progress == 0:
                        print(count)
            component[new_node].append(flip)
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
