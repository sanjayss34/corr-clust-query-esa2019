import Queue
from copy import deepcopy
import itertools
import numpy as np
import scipy.optimize
from gurobipy import *
np.set_printoptions(threshold=100000)

def three_approx(adj_mat, vertices, random_gen):
    if len(vertices) == 0:
        return set()
    pivot = random_gen.choice(list(vertices), 1)[0]
    cluster = {pivot}
    for v in vertices:
        if adj_mat[v][pivot] == 1:
            cluster.add(v)
    new_vertices = vertices-cluster
    clusters = three_approx(adj_mat, new_vertices, random_gen)
    clusters.add(frozenset(cluster))
    return clusters

def random_query_pivot(adj_mat, oracle_mat, vertices, p, random_gen):
    if len(vertices) == 0:
        return set(), np.zeros_like(adj_mat, dtype=np.int32)
    pivot = random_gen.choice(list(vertices), 1)[0]
    cluster = {pivot}
    for v in vertices:
        if adj_mat[v][pivot] == 1:
            cluster.add(v)
    queried = np.zeros((adj_mat.shape[0], adj_mat.shape[1]), dtype=np.int32)
    for v in vertices-{pivot}:
        for w in vertices-{pivot, v}:
            if (adj_mat[pivot][v] == 1 and adj_mat[pivot][w] == 1 and adj_mat[v][w] == -1) or (adj_mat[pivot][v] == 1 and adj_mat[pivot][w] == -1 and adj_mat[v][w] == 1) or (adj_mat[pivot][v] == -1 and adj_mat[pivot][w] == 1 and adj_mat[v][w] == 1):
                if random_gen.rand() < p:
                    if adj_mat[pivot][v] == 1:
                        if oracle_mat[pivot][v] == -1:
                            cluster -= {v}
                        queried[pivot][v] = 1
                        queried[v][pivot] = 1
                    if adj_mat[pivot][w] == 1 or oracle_mat[pivot][v] == 1:
                        if adj_mat[pivot][w] == -1 and oracle_mat[pivot][w] == 1:
                            cluster.add(w)
                        elif adj_mat[pivot][w] == 1 and oracle_mat[pivot][w] == -1:
                            cluster -= {w}
                        queried[pivot][w] = 1
                        queried[w][pivot] = 1
    new_vertices = vertices-cluster
    clusters, full_queried = random_query_pivot(adj_mat, oracle_mat, new_vertices, p, random_gen)
    for u in vertices:
        for v in vertices:
            full_queried[u][v] = max(full_queried[u][v], queried[u][v])
    clusters.add(frozenset(cluster))
    return clusters, full_queried

def query_pivot(adj_mat, oracle_mat, vertices):
    if len(vertices) == 0:
        return set(), np.zeros_like(adj_mat, dtype=np.int32)
    pivot = list(vertices)[0]
    cluster = {pivot}
    for v in vertices:
        if adj_mat[v][pivot] == 1:
            cluster.add(v)
    queried = np.zeros_like(adj_mat, dtype=np.int32)
    for v in vertices-{pivot}:
        for w in vertices-{pivot, v}:
            if (adj_mat[pivot][v] == 1 and adj_mat[pivot][w] == 1 and adj_mat[v][w] == -1) or (adj_mat[pivot][v] == 1 and adj_mat[pivot][w] == -1 and adj_mat[v][w] == 1) or (adj_mat[pivot][v] == -1 and adj_mat[pivot][w] == 1 and adj_mat[v][w] == 1):
                if queried[pivot][v] == 1 and queried[pivot][w] == 1:
                    continue
                if queried[pivot][v] == 1:
                    if oracle_mat[pivot][v] != adj_mat[pivot][v]:
                        continue
                    else:
                        queried[pivot][w] = 1
                        queried[w][pivot] = 1
                        if oracle_mat[pivot][w] != adj_mat[pivot][w]:
                            if adj_mat[pivot][w] == 1:
                                cluster -= {w}
                            else:
                                cluster.add(w)
                elif queried[pivot][w] == 1:
                    if oracle_mat[pivot][w] != adj_mat[pivot][w]:
                        continue
                    else:
                        queried[pivot][v] = 1
                        queried[v][pivot] = 1
                        if oracle_mat[pivot][v] != adj_mat[pivot][v]:
                            if adj_mat[pivot][v] == 1:
                                cluster -= {v}
                            else:
                                cluster.add(v)
                else:
                    queried[pivot][v] = 1
                    queried[v][pivot] = 1
                    if oracle_mat[pivot][v] != adj_mat[pivot][v]:
                        if adj_mat[pivot][v] == 1:
                            cluster -= {v}
                        else:
                            cluster.add(v)
                        continue
                    else:
                        queried[pivot][w] = 1
                        queried[w][pivot] = 1
                        if oracle_mat[pivot][w] != adj_mat[pivot][w]:
                            if adj_mat[pivot][w] == 1:
                                cluster -= {w}
                            else:
                                cluster.add(w)
    new_vertices = vertices-cluster
    clusters, full_queried = query_pivot(adj_mat, oracle_mat, new_vertices)
    for u in vertices:
        for v in vertices:
            full_queried[u][v] = max(full_queried[u][v], queried[u][v])
    clusters.add(frozenset(cluster))
    return clusters, full_queried

def _bocker_caseB(new_adj_mat, parent_vertices):
    for x in parent_vertices:
        for y in parent_vertices:
            if x == y or new_adj_mat[x][y] <= 0:
                continue
            for z in parent_vertices:
                if z in {x, y} or new_adj_mat[x][z] <= 0 or new_adj_mat[y][z] <= 0:
                    continue
                for v1 in parent_vertices:
                    if v1 in {x, y, z} or not ((new_adj_mat[x][v1] < 0 and new_adj_mat[y][v1] > 0) or (new_adj_mat[y][v1] < 0 and new_adj_mat[x][v1] > 0) or (new_adj_mat[x][v1] == 0 and new_adj_mat[y][v1] == 0) or ((new_adj_mat[y][v1] == 0 or new_adj_mat[x][v1] == 0) and min(new_adj_mat[x][v1], new_adj_mat[y][v1]) < 0 and new_adj_mat[z][v1] >= 0) or ((new_adj_mat[x][v1] == 0 or new_adj_mat[y][v1] == 0) and max(new_adj_mat[x][v1], new_adj_mat[y][v1]) > 0 and new_adj_mat[z][v1] <= 0)):
                        continue
                    for v2 in parent_vertices:
                        if v2 in {x, y, z, v1} or not ((new_adj_mat[x][v2] < 0 and new_adj_mat[y][v2] > 0) or (new_adj_mat[y][v2] < 0 and new_adj_mat[x][v2] > 0) or (new_adj_mat[x][v2] == 0 and new_adj_mat[y][v2] == 0) or ((new_adj_mat[y][v2] == 0 or new_adj_mat[x][v2] == 0) and min(new_adj_mat[x][v2], new_adj_mat[y][v2]) < 0 and new_adj_mat[z][v2] >= 0) or ((new_adj_mat[x][v2] == 0 or new_adj_mat[y][v2] == 0) and max(new_adj_mat[x][v2], new_adj_mat[y][v2]) > 0 and new_adj_mat[z][v2] <= 0)):
                            continue
                        return (x, y)
    return None

def _bocker_path(adj_mat, path):
    K = []
    for _ in range(len(path)):
        K.append([])
        for _ in range(len(path)):
            K[-1].append(0)
    for j in range(len(path)):
        for i in range(j-2, -1, -1):
            K[i][j] = K[i][j-1]+K[i+1][j]-K[i+1][j-1]-adj_mat[path[i]][path[j]]
    best_i = [-1 for _ in path]
    D = {-1: 0}
    S = {-1: 0}
    for j in range(len(path)-1):
        S[j] = adj_mat[path[j]][path[j+1]]
    for j in range(len(path)):
        min_cost = np.inf
        best_index = -1
        for i in range(-1, j):
            if D[i]+S[i]+K[i+1][j] < min_cost:
                min_cost = D[i]+S[i]+K[i+1][j]
                best_index = i
        best_i[j] = best_index
        D[j] = min_cost
    j = len(path)-1
    while j >= 0:
        for i1 in range(best_i[j]+1, j+1):
            for i2 in range(best_i[j]+1, i1):
                adj_mat[path[i1]][path[i2]] = 1
                adj_mat[path[i2]][path[i1]] = adj_mat[path[i1]][path[i2]]
        if best_i[j] >= 0:
            adj_mat[path[best_i[j]]][path[best_i[j]+1]] = -1
            adj_mat[path[best_i[j]+1]][path[best_i[j]]] = -1
        j = best_i[j]
    return adj_mat, D[-1]

def _bocker_min_cut(adj_mat, vertices, pair):
    s, t = pair
    n = len(adj_mat)
    capacity_mat = deepcopy(adj_mat)
    for i in vertices:
        for j in vertices:
            if j < i and capacity_mat[i][j] < 0:
                capacity_mat[i][j] = 0
                capacity_mat[j][i] = 0
    flow_mat = np.zeros((n, n))
    reachable_vertices = set()
    terminate_flag = False
    while True:
        parents = [-1 for _ in range(n)]
        flow_values = [np.inf for _ in range(n)]
        q = Queue.Queue()
        q.put(s)
        reachable_vertices = {s}
        while not q.empty():
            u = q.get()
            for v in vertices:
                if capacity_mat[u][v]-flow_mat[u][v] > 0:
                    if v not in reachable_vertices:
                        q.put(v)
                        reachable_vertices.add(v)
                        parents[v] = u
                        flow_values[v] = int(min(flow_values[u], capacity_mat[u][v]-flow_mat[u][v]))
        if parents[t] > -1:
            v = t
            while v != s:
                flow_mat[parents[v]][v] = flow_values[t]
                flow_mat[v][parents[v]] = -flow_values[t]
                v = parents[v]
        elif terminate_flag:
            break
        else:
            terminate_flag = True
    unreachable_vertices = vertices-reachable_vertices
    cost = 0
    for u in reachable_vertices:
        for v in unreachable_vertices:
            if adj_mat[u][v] > 0:
                cost += adj_mat[u][v]
            adj_mat[u][v] = -abs(adj_mat[u][v])
            adj_mat[v][u] = -abs(adj_mat[u][v])
    return adj_mat, cost

def _bocker_partition_set(s):
    if len(s) == 0:
        result = set()
        result.add(frozenset(set()))
        return result
    element = list(s)[0]
    others = s-{element}
    clusterings = set()
    for size in range(len(s)):
        subsets = itertools.combinations(others, size)
        for subset in subsets:
            cluster = {element}.union(subset)
            results = _bocker_partition_set(s-cluster)
            for clustering in results:
                full_clustering = frozenset(set(clustering).union({frozenset(cluster)}))
                clusterings.add(full_clustering)
    return clusterings

def _bocker_remove_cliques(vertices, adj_mat):
    new_vertices = deepcopy(vertices)
    visited_map = {u: False for u in vertices}
    for u in vertices:
        if visited_map[u]:
            continue
        stack = [u]
        component = {u}
        visited_map[u] = True
        while len(stack) > 0:
            v = stack.pop()
            for w in vertices:
                if adj_mat[v][w] > 0 and not visited_map[w]:
                    component.add(w)
                    visited_map[w] = True
                    stack.append(w)
        if not any(adj_mat[u][v] <= 0 for u in component for v in component-{u}):
            new_vertices -= component
    return new_vertices

def bocker_alg(adj_mat, oracle_mat):
    n = len(adj_mat)
    parent = range(n)
    new_adj_mat = deepcopy(adj_mat)
    queries = 0
    threshold_delete_cost = 1
    threshold_merge_cost = 1.5
    max_cost = max(threshold_delete_cost, threshold_merge_cost)
    min_cost = min(threshold_delete_cost, threshold_merge_cost)
    characteristic_f = lambda x: x**max_cost-x**(max_cost-min_cost)-1
    fprime = lambda x: max_cost*x**(max_cost-1) - (max_cost-min_cost)*x**(max_cost-min_cost-1)
    newton_maxiter = 100000
    threshold = scipy.optimize.newton(characteristic_f, x0=2.0, fprime=fprime, maxiter=newton_maxiter)
    parent_vertices = set(range(n))
    while True:
        parent_vertices = _bocker_remove_cliques(parent_vertices, new_adj_mat)
        best_edge = None
        best_branching_number = np.inf
        best_costs = None
        conflict_triple_exists = False
        for u in parent_vertices:
            for v in parent_vertices:
                if u == v or new_adj_mat[u][v] <= 0:
                    continue
                merge_cost = 0
                for w in parent_vertices:
                    if parent[w] != w or w == u or w == v:
                        continue
                    if (new_adj_mat[u][w] < 0 and new_adj_mat[v][w] > 0) or (new_adj_mat[u][w] > 0 and new_adj_mat[v][w] < 0):
                        merge_cost += min(abs(new_adj_mat[u][w]), abs(new_adj_mat[v][w]))
                        if abs(new_adj_mat[u][w]) == abs(new_adj_mat[v][w]):
                            merge_cost -= 0.5
                        conflict_triple_exists = True
                    elif new_adj_mat[u][w] == 0 or new_adj_mat[v][w] == 0:
                        merge_cost += 0.5
                delete_cost = new_adj_mat[u][v]
                max_cost = max(merge_cost, delete_cost)
                if delete_cost >= threshold_delete_cost and merge_cost >= threshold_merge_cost:
                    best_edge = [u, v]
                    best_branching_number = threshold/2
                    best_costs = (delete_cost, merge_cost)
                    break
                if min(merge_cost, delete_cost) == 0:
                    continue
                characteristic_f = lambda x: x**max_cost-x**(max_cost-delete_cost)-x**(max_cost-merge_cost)
                fprime = lambda x: max_cost*x**(max_cost-1)-(max_cost-delete_cost)*x**(max_cost-delete_cost-1)-(max_cost-merge_cost)*x**(max_cost-merge_cost-1)
                sol1 = scipy.optimize.newton(characteristic_f, x0=2.0, fprime=fprime, maxiter=newton_maxiter)
                if sol1 < best_branching_number:
                    best_edge = [u, v]
                    best_branching_number = sol1
                    best_costs = (delete_cost, merge_cost)
                    if best_branching_number <= threshold:
                        break
            if best_branching_number <= threshold:
                break
        if best_branching_number > threshold:
            best_edge = _bocker_caseB(new_adj_mat, parent_vertices)
            best_costs = 'caseB'
            if best_edge is None:
                break
        queries += 1
        if oracle_mat[best_edge[0]][best_edge[1]] == 1:
            parent[best_edge[1]] = best_edge[0]
            for w in parent_vertices:
                if parent[w] == w and w not in best_edge:
                    new_adj_mat[w][best_edge[0]] = new_adj_mat[w][best_edge[0]]+new_adj_mat[w][best_edge[1]]
                    new_adj_mat[best_edge[0]][w] = new_adj_mat[w][best_edge[0]]
            parent_vertices.remove(best_edge[1])
        else:
            new_adj_mat[best_edge[0]][best_edge[1]] = -n*(n-1)/2-1
            new_adj_mat[best_edge[1]][best_edge[0]] = -n*(n-1)/2-1
    clusters = set()
    clustered = [-1]*n
    children = [[] for _ in range(n)]
    connected_components = set()
    for u in parent_vertices:
        stack = [u]
        component = {u}
        visited_map = {u: True}
        while len(stack) > 0:
            v = stack.pop()
            for w in parent_vertices:
                if new_adj_mat[v][w] > 0 and w not in visited_map:
                    component.add(w)
                    visited_map[w] = True
                    stack.append(w)
        connected_components.add(frozenset(component))
    for component in connected_components:
        parents = list(component)
        if len(parents) in {3, 4}:
            all_clusterings = _bocker_partition_set(component)
            min_cost = np.inf
            vertex_map = {p: i for i, p in enumerate(parents)}
            best_clustering = None
            for clustering in all_clusterings:
                clustering_mapped = set()
                for cluster in clustering:
                    clustering_mapped.add(frozenset(vertex_map[v] for v in cluster))
                cost = count_mistakes(clustering_mapped, new_adj_mat[[[v] for v in parents],parents], bocker_weighted=True)
                if cost < min_cost:
                    min_cost = cost
                    best_clustering = clustering
            for clust in best_clustering:
                for u1 in clust:
                    for u2 in clust:
                        new_adj_mat[u1][u2] = 1
                        new_adj_mat[u2][u1] = 1
                for clust2 in best_clustering-{clust}:
                    for u1 in clust:
                        for u2 in clust2:
                            new_adj_mat[u1][u2] = -1
                            new_adj_mat[u2][u1] = -1
        elif len(parents) <= 2:
            pass
        neg_edges = set()
        for i in range(len(parents)):
            for j in range(i):
                if new_adj_mat[parents[i]][parents[j]] < 0:
                    neg_edges.add((parents[i], parents[j]))
        # Check for clique case
        if len(parents) > 4 and len(neg_edges) == 0:
            for i in range(len(parents)):
                for j in range(i):
                    new_adj_mat[parents[i]][parents[j]] = 1
                    new_adj_mat[parents[j]][parents[i]] = 1
        elif len(parents) > 4 and len(neg_edges) == 1:
            u, v = list(neg_edges)[0]
            adj_mat_min_cut = deepcopy(new_adj_mat)
            adj_mat_min_cut, min_cut_cost = _bocker_min_cut(adj_mat_min_cut, set(parents), [u, v])
            if min_cut_cost < -new_adj_mat[u][v]:
                new_adj_mat = adj_mat_min_cut
            else:
                for i in range(len(parents)):
                    for j in range(i):
                        new_adj_mat[parents[i]][parents[j]] = 1
                        new_adj_mat[parents[j]][parents[i]] = 1
        elif len(parents) > 4:
            # Checking for path
            degree1_vert = None
            degrees = []
            max_degree = 0
            for v in parents:
                deg = 0
                for u in parents:
                    if u != v and new_adj_mat[u][v] > 0:
                        deg += 1
                if deg == 1:
                    degree1_vert = v
                max_degree = max(deg, max_degree)
                degrees.append(deg)
            path_flag = False
            if degree1_vert is not None:
                # Construct path
                path = [degree1_vert]
                vertices_in_path = set(path)
                while len(path) < len(parents):
                    v = path[-1]
                    neighbor = None
                    for u in parents:
                        if new_adj_mat[u][v] > 0 and u not in vertices_in_path:
                            vertices_in_path.add(u)
                            path.append(u)
                            neighbor = u
                            break
                    if neighbor is None:
                        break
                if len(path) == len(parents):
                    path_flag = True
                    new_adj_mat, _ = _bocker_path(new_adj_mat, path)
            if not path_flag:
                circle_flag = False
                # Check circle case
                if max_degree == 2:
                    circle_path = [parents[0]]
                    in_circle = [0 for _ in range(n)]
                    in_circle[0] = 1
                    while len(circle_path) < len(parents):
                        cur_length = len(circle_path)
                        for u in parents:
                            if in_circle[u] == 0 and new_adj_mat[circle_path[-1]][u] > 0:
                                circle_path.append(u)
                                in_circle[u] = 1
                                break
                        if cur_length == len(circle_path):
                            break
                    if len(circle_path) == len(parents) and new_adj_mat[circle_path[0]][circle_path[-1]] > 0:
                        circle_flag = True
                        min_cost = np.inf
                        best_adj_mat = None
                        for j in range(-1, len(parents)-1):
                            edge_remove_adj_mat = deepcopy(new_adj_mat)
                            edge_remove_adj_mat[circle_path[j]][circle_path[j+1]] *= -1
                            edge_remove_adj_mat[circle_path[j+1]][circle_path[j]] *= -1
                            path = circle_path[j+1:]+circle_path[:j+1]
                            edge_remove_adj_mat, cost = _bocker_path(edge_remove_adj_mat, path)
                            if cost < min_cost:
                                min_cost = cost
                                best_adj_mat = edge_remove_adj_mat
                        # clique cost
                        clique_cost = 0
                        clique_adj_mat = deepcopy(new_adj_mat)
                        for u in parents:
                            for v in parents:
                                if u > v:
                                    if new_adj_mat[u][v] < 0:
                                        clique_cost -= new_adj_mat[u][v]
                                    clique_adj_mat[u][v] = abs(clique_adj_mat[u][v])
                                    clique_adj_mat[v][u] = clique_adj_mat[u][v]
                        if min_cost < clique_cost:
                            new_adj_mat = best_adj_mat
                        else:
                            new_adj_mat = clique_adj_mat
                if not circle_flag:
                    print('ERROR! NONE OF THE END CASES')
                    exit()
    for i in range(n):
        if parent[i] != i:
            children[parent[i]].append(i)
    for i in range(n):
        if parent[i] != i or clustered[i] > -1:
            continue
        stack = [i]+[j for j in range(n) if parent[j] == j and new_adj_mat[i][j] > 0]
        cluster = set()
        while len(stack) > 0:
            v = stack.pop()
            clustered[v] = i
            cluster.add(v)
            stack += [u for u in children[v] if clustered[u] == -1]
        clusters.add(frozenset(cluster))
    return clusters, queries

def delta_good(vertex, clust, vertices, adj_mat, delta):
    plus_intersect_clust = set()
    for u in clust:
        if adj_mat[vertex][u] == 1 or u == vertex:
            plus_intersect_clust.add(u)
    v_minus_clust = vertices-clust
    plus_intersect_complement = set()
    for u in v_minus_clust:
        if adj_mat[vertex][u] == 1 or u == vertex:
            plus_intersect_complement.add(u)
    if (len(plus_intersect_clust) >= (1-delta)*len(clust)) and (len(plus_intersect_complement) <= delta*len(clust)):
        return True
    return False

# Algorithm of Bansal, Blum, Chawla (FOCS 2004)
def cautious(adj_mat, vertices):
    if len(vertices) < 1:
        return set()
    v = vertices.pop()
    vertices.add(v)
    A = {v}
    delta = 1.0/8
    for u in vertices:
        if adj_mat[v][u] == 1:
            A.add(u)
    while True:
        remove_u = None
        for u in A:
            if not delta_good(u, A, vertices, adj_mat, 3*delta):
                remove_u = u
                break
        if remove_u is not None:
            A.remove(remove_u)
        else:
            break
    Y_add = set()
    for u in vertices:
        if delta_good(u, A, vertices, adj_mat, 7*delta):
            Y_add.add(u)
    A = A.union(Y_add)
    if len(A) == 0:
        return set(frozenset([v]) for v in vertices)
    return {frozenset(A)}.union(cautious(adj_mat, vertices-A))

def _lp_pivot(adj_mat, vertices, lp, random_gen):
    if len(vertices) == 0:
        return set()
    pivot = random_gen.choice(list(vertices), 1)[0]
    C = {pivot}
    a = 0.19
    b = 0.5095
    for v in vertices-{pivot}:
        if adj_mat[pivot][v] == 1:
            p_uv = ((lp[pivot][v]-a)/(b-a))**2
            if lp[pivot][v] < a:
                p_uv = 0.0
            elif lp[pivot][v] >= b:
                p_uv = 1.0
            if random_gen.rand() < 1-p_uv:
                C.add(v)
        else:
            if random_gen.rand() < 1-lp[pivot][v]:
                C.add(v)
    new_vertices = vertices-C
    clusters = _lp_pivot(adj_mat, new_vertices, lp, random_gen)
    clusters.add(frozenset(C))
    return clusters

def lp_rounding(adj_mat, random_gen, ilp=False):
    m = Model("lp")
    if ilp:
        m = Model("mip")
    m.setParam('OutputFlag', False)
    lp_vars = []
    n = len(adj_mat)
    for i in range(n):
        lp_vars.append([])
        for j in range(i):
            if ilp:
                lp_vars[-1].append(m.addVar(vtype=GRB.BINARY, name=str(i)+','+str(j)))
            else:
                lp_vars[-1].append(m.addVar(vtype=GRB.CONTINUOUS, name=str(i)+','+str(j)))
    def obj():
        res = 0
        for i in range(n):
            for j in range(i):
                if adj_mat[i][j] <= 0:
                    res += 1-lp_vars[i][j]
                else:
                    res += lp_vars[i][j]
        return res
    m.setObjective(obj(), GRB.MINIMIZE)
    m.update()
    count = 0
    for i in range(n):
        for j in range(i):
            for k in range(j):
                m.addLConstr(lp_vars[i][j]+lp_vars[j][k] >= lp_vars[i][k])
                m.addLConstr(lp_vars[i][j]+lp_vars[i][k] >= lp_vars[j][k])
                m.addLConstr(lp_vars[i][k]+lp_vars[j][k] >= lp_vars[i][j])
                count += 3
            m.addLConstr(lp_vars[i][j] >= 0.0)
            m.addLConstr(lp_vars[i][j] <= 1.0)
            count += 2
    m.optimize()
    lp = -1*np.ones((n, n))
    if ilp:
        for i in range(n):
            for j in range(i):
                lp[i][j] = 2*(1-lp_vars[i][j].x)-1
                lp[j][i] = 2*(1-lp_vars[i][j].x)-1
        clusters, _ = query_pivot(lp, lp, set(range(n)))
    else:
        for i in range(n):
            for j in range(i):
                lp[i][j] = lp_vars[i][j].x
                lp[j][i] = lp_vars[i][j].x
        clusters = _lp_pivot(adj_mat, set(range(n)), lp, random_gen)
    return clusters

def _lp_pivot_weighted(adj_mat, vertices, lp, random_gen):
    if len(vertices) == 0:
        return set()
    pivot = random_gen.choice(list(vertices), 1)[0]
    C = {pivot}
    a = 0.19
    b = 0.5095
    for v in vertices-{pivot}:
        if random_gen.rand() < adj_mat[pivot][v]:
            p_uv = ((lp[pivot][v]-a)/(b-a))**2
            if lp[pivot][v] < a:
                p_uv = 0.0
            elif lp[pivot][v] >= b:
                p_uv = 1.0
            if random_gen.rand() < 1-p_uv:
                C.add(v)
        else:
            if random_gen.rand() < 1-lp[pivot][v]:
                C.add(v)
    new_vertices = vertices-C
    clusters = _lp_pivot_weighted(adj_mat, new_vertices, lp, random_gen)
    clusters.add(frozenset(C))
    return clusters

def lp_rounding_weighted(adj_mat, random_gen, ilp=False):
    m = Model("lp")
    if ilp:
        m = Model("mip")
    m.setParam('OutputFlag', False)
    lp_vars = []
    n = len(adj_mat)
    for i in range(n):
        lp_vars.append([])
        for j in range(i):
            if ilp:
                lp_vars[-1].append(m.addVar(vtype=GRB.BINARY, name=str(i)+','+str(j)))
            else:
                lp_vars[-1].append(m.addVar(vtype=GRB.CONTINUOUS, name=str(i)+','+str(j)))
    def obj():
        res = 0
        for i in range(n):
            for j in range(i):
                res += adj_mat[i][j]*lp_vars[i][j]+(1.0-adj_mat[i][j])*(1-lp_vars[i][j])
        return res
    m.setObjective(obj(), GRB.MINIMIZE)
    m.update()
    count = 0
    for i in range(n):
        for j in range(i):
            for k in range(j):
                m.addLConstr(lp_vars[i][j]+lp_vars[j][k] >= lp_vars[i][k])
                m.addLConstr(lp_vars[i][j]+lp_vars[i][k] >= lp_vars[j][k])
                m.addLConstr(lp_vars[i][k]+lp_vars[j][k] >= lp_vars[i][j])
                count += 3
            m.addLConstr(lp_vars[i][j] >= 0.0)
            m.addLConstr(lp_vars[i][j] <= 1.0)
            count += 2
    m.optimize()
    lp = -1*np.ones((n, n))
    if ilp:
        for i in range(n):
            for j in range(i):
                lp[i][j] = 2*(1-lp_vars[i][j].x)-1
                lp[j][i] = 2*(1-lp_vars[i][j].x)-1
        clusters, _ = query_pivot(lp, lp, set(range(n)))
    else:
        for i in range(n):
            for j in range(i):
                lp[i][j] = lp_vars[i][j].x
                lp[j][i] = lp_vars[i][j].x
        clusters = _lp_pivot_weighted(adj_mat, set(range(n)), lp, random_gen)
    return clusters

def count_mistakes(clusters, adj_mat, weighted=False, bocker_weighted=False):
    n = len(adj_mat)
    output_adj_mat = -1*np.ones((n, n), dtype=np.int32)
    for clust in clusters:
        for i in clust:
            for j in clust:
                output_adj_mat[i][j] = 1
    mistakes = 0
    for i in range(n):
        for j in range(n):
            if weighted:
                mistakes += np.abs(adj_mat[i][j]-(output_adj_mat[i][j]+1)/2)
            elif bocker_weighted:
                if (output_adj_mat[i][j] > 0 and adj_mat[i][j] < 0) or (output_adj_mat[i][j] < 0 and adj_mat[i][j] > 0):
                    mistakes += abs(adj_mat[i][j])
            else:
                if output_adj_mat[i][j] != adj_mat[i][j]:
                    mistakes += 1
    mistakes /= 2
    return mistakes
