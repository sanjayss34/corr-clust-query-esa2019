import sys
import numpy as np
from copy import deepcopy
import time
from gurobipy import *
from algorithms import *
np.random.seed(1666)

if __name__ == '__main__':
    dataset = sys.argv[1]
    if len(sys.argv) > 2:
        edge_choice = int(sys.argv[2])
    n = 0
    adj_mat = None
    oracle_mat = None
    if dataset in {'skew', 'sqrt'}:
        graph_f = open('data/'+dataset+'/graph.txt')
        graph_lines = graph_f.readlines()
        gold_f = open('data/'+dataset+'/gold.txt')
        gold_lines = gold_f.readlines()
        n = int(graph_lines[0].split()[0])
        adj_mat = np.zeros((n, n), dtype=np.int32)
        oracle_mat = -1*np.ones((n, n), dtype=np.int32)
        for i in range(1, len(graph_lines)-1):
            parts = graph_lines[i].split()
            v1 = int(parts[0])
            v2 = int(parts[1])
            label = int(float(parts[2]))
            adj_mat[v1][v2] = 2*label-1
            adj_mat[v2][v1] = adj_mat[v1][v2]
        for v in range(n):
            adj_mat[v][v] = 1
        L = int(0.1*n*(n-1)/2)
        oracle_mat = deepcopy(adj_mat)
        clusters = {}
        for line in gold_lines:
            parts = line.split()
            if int(parts[1]) not in clusters:
                clusters[int(parts[1])] = []
            clusters[int(parts[1])].append(int(parts[0]))
        if edge_choice == 1: # Edges chosen uniformly at random
            count = 0
            flipped_set = set()
            while count < L:
                row = np.random.randint(0, high=adj_mat.shape[0])
                col = np.random.randint(0, high=adj_mat.shape[1])
                if row != col and (row, col) not in flipped_set and (col, row) not in flipped_set:
                    adj_mat[row][col] *= -1
                    adj_mat[col][row] *= -1
                    count += 1
                    flipped_set.add((row, col))
        elif edge_choice == 2: # L/num_cliques chosen from each cluster
            num_flipped_edges = 0
            for c in clusters:
                count = 0
                flipped_set = set()
                while count < min(L/len(clusters), len(clusters[c])-1):
                    row = np.random.choice(clusters[c], 1)[0]
                    col = np.random.choice(clusters[c], 1)[0]
                    if row != col and (row, col) not in flipped_set and (col, row) not in flipped_set:
                        adj_mat[row][col] *= -1
                        adj_mat[col][row] *= -1
                        count += 1
                        flipped_set.add((row, col))
                num_flipped_edges += count
            L = num_flipped_edges
        elif edge_choice == 3:
            num_flipped_edges = 0
            for c in clusters:
                count = 0
                flipped_set = set()
                while count < min(L/len(clusters), len(clusters[c])-1):
                    row = np.random.choice(clusters[c], 1)[0]
                    col = np.random.choice(clusters[c], 1)[0]
                    if row != col and (row, col) not in flipped_set and (col, row) not in flipped_set:
                        adj_mat[row][col] *= -1
                        adj_mat[col][row] *= -1
                        count += 1
                        flipped_set.add((row, col))
                num_flipped_edges += count
            L = num_flipped_edges
            num_flipped_edges = 0
            for c in clusters:
                for c2 in clusters:
                    if c == c2:
                        continue
                    count = 0
                    num_to_flip = int(math.ceil(0.01*len(clusters[c])*len(clusters[c2])))
                    flipped_set = set()
                    while count < num_to_flip:
                        row = np.random.choice(clusters[c], 1)[0]
                        col = np.random.choice(clusters[c2], 1)[0]
                        if (row, col) not in flipped_set and (col, row) not in flipped_set:
                            adj_mat[row][col] *= -1
                            adj_mat[col][row] *= -1
                            count += 1
                            flipped_set.add((row, col))
                    num_flipped_edges += count
            L += num_flipped_edges
    crowd_mat = None
    weighted_adj_mat = None
    if dataset in {'cora', 'landmarks', 'allsports'}:
        graph_f = open('data/'+dataset+'/graph.txt')
        graph_lines = graph_f.readlines()
        gold_f = open('data/'+dataset+'/gold.txt')
        gold_lines = gold_f.readlines()
        n = int(graph_lines[0].split()[0])
        adj_mat = -1*np.ones((n, n), dtype=np.int32)
        weighted_adj_mat = -1*np.ones((n, n), dtype=np.float32)
        oracle_mat = -1*np.ones((n, n), dtype=np.int32)
        for i in range(1, len(graph_lines)-1):
            parts = graph_lines[i].split()
            v1 = int(parts[0])
            v2 = int(parts[1])
            weight = float(parts[2])
            """adj_mat[v1][v2] = weight
            adj_mat[v2][v1] = adj_mat[v1][v2]"""
            if weight >= 0.5:
                adj_mat[v1][v2] = 1
            adj_mat[v2][v1] = adj_mat[v1][v2]
            weighted_adj_mat[v1][v2] = weight
            weighted_adj_mat[v2][v1] = weight
        for v in range(n):
            adj_mat[v][v] = 1
        clusters = {}
        for line in gold_lines:
            parts = line.split()
            if int(parts[1]) not in clusters:
                clusters[int(parts[1])] = []
            clusters[int(parts[1])].append(int(parts[0]))
        for clust in clusters.values():
            for i in clust:
                for j in clust:
                    oracle_mat[i][j] = 1
        if dataset != 'cora':
            crowd_f = open('data/'+dataset+'/answers.txt')
            crowd_lines = crowd_f.readlines()
            crowd_mat = -1*np.ones((n, n))
            for i, line in enumerate(crowd_lines):
                parts = line.split()
                u = int(parts[0])
                v = int(parts[1])
                answer = float(sum([int(parts[j]) for j in range(2, len(parts))]))/(len(parts)-2)
                if answer >= 0.5:
                    crowd_mat[u][v] = 1
                crowd_mat[v][u] = crowd_mat[u][v]
            crowd_mistakes = 0
            for i in range(n):
                for j in range(n):
                    if crowd_mat[i][j] != oracle_mat[i][j]:
                        crowd_mistakes += 1
            crowd_mistakes /= 2
    if dataset == 'gym':
        graph_f = open('data/gym/graph.txt')
        graph_lines = graph_f.readlines()
        gold_f = open('data/gym/gold.txt')
        gold_lines = gold_f.readlines()
        crowd_f = open('data/gym/answers.txt')
        node_map = {}
        n = 94
        oracle_mat = -1*np.ones((n, n), dtype=np.int32)
        count = 0
        for line in gold_lines:
            parts = line.split(',')
            clust = []
            for i in parts[1:]:
                node_map[int(i)] = count
                clust.append(count)
                count += 1
            for i in clust:
                for j in clust:
                    oracle_mat[i][j] = 1
        adj_mat = -1*np.ones((n, n), dtype=np.int32)
        weighted_adj_mat = -1*np.ones((n, n), dtype=np.float32)
        for i in range(1, len(graph_lines)-1):
            parts = graph_lines[i].split(',')
            v1 = node_map[int(parts[0])]
            v2 = node_map[int(parts[1])]
            weight = float(parts[2])
            """adj_mat[v1][v2] = weight
            adj_mat[v2][v1] = adj_mat[v1][v2]"""
            if weight >= 0.5:
                adj_mat[v1][v2] = 1
            adj_mat[v2][v1] = adj_mat[v1][v2]
            weighted_adj_mat[v1][v2] = weight
            weighted_adj_mat[v2][v1] = weight
        for v in range(n):
            adj_mat[v][v] = 1
        crowd_lines = crowd_f.readlines()
        crowd_mat = -1*np.ones((n, n))
        for i, line in enumerate(crowd_lines):
            parts = line.split(',')
            u = int(parts[0])
            v = int(parts[1])
            answer = float(sum([int(parts[j]) for j in range(2, len(parts))]))/(len(parts)-2)
            if answer >= 0.5:
                crowd_mat[node_map[u]][node_map[v]] = 1
            crowd_mat[node_map[v]][node_map[u]] = crowd_mat[node_map[u]][node_map[v]]
    print('Loaded data')
    L = 0
    for i in range(n):
        for j in range(n):
            if adj_mat[i][j] != oracle_mat[i][j]:
                L += 1
    L /= 2
    weighted_adj_mat = deepcopy(adj_mat)
    gold_mat = deepcopy(oracle_mat)
    if dataset in {'gym', 'landmarks', 'allsports'} and int(sys.argv[2]) == 1:
        oracle_mat = deepcopy(crowd_mat)
    if dataset in {'gym', 'landmarks', 'allsports'} and int(sys.argv[2]) == 2:
        clusters = lp_rounding(adj_mat, np.random, ilp=True)
        ilp_mat = -1*np.ones((n, n), dtype=np.int32)
        for clust in clusters:
            for i in clust:
                for j in clust:
                    ilp_mat[i][j] = 1
        mistakes = count_mistakes(clusters, adj_mat)
        oracle_mat = deepcopy(ilp_mat)
        gold_mat = deepcopy(adj_mat)

    clusters = cautious(adj_mat, set(range(n)))
    start = time.time()
    mistakes = count_mistakes(clusters, gold_mat)
    end = time.time()
    print('Cautious (BBC)')
    print('Mistakes: ' + str(mistakes))
    print('Time: ' + str(end-start))
    sys.stdout.flush()

    print('Bocker')
    start = time.time()
    clusters, num_queries = bocker_alg(adj_mat, oracle_mat)
    end = time.time()
    alg_time = end-start
    mistakes = count_mistakes(clusters, gold_mat)
    print('Mistakes: ' + str(mistakes))
    print('Queries: ' + str(num_queries))
    print('Time: ' + str(alg_time))

    mistakes_trials = []
    times_trials = []
    print('3-approximation Pivot (ACN)')
    for trial in range(3):
        start = time.time()
        clusters = three_approx(adj_mat, set(range(n)), np.random)
        end = time.time()
        times_trials.append(end-start)
        mistakes = count_mistakes(clusters, gold_mat)
        mistakes_trials.append(mistakes)
        print('Trial ' + str(trial) + ' finished')
    print('Mistakes: ' + str(np.mean(mistakes_trials)))
    print('Avg. Time: ' + str(np.mean(times_trials)))

    print('Query pivot')
    start = time.time()
    clusters, queried = query_pivot(adj_mat, oracle_mat, set(range(n)))
    end = time.time()
    alg_time = end-start
    num_queries = np.sum(queried)/2
    mistakes = count_mistakes(clusters, gold_mat)
    print('Mistakes: ' + str(mistakes))
    print('Queries: ' + str(num_queries))
    print('Time: ' + str(alg_time))

    queries_trials = []
    mistakes_trials = []
    times_trials = []
    for trial in range(3):
        start = time.time()
        clusters, queried = random_query_pivot(adj_mat, oracle_mat, set(range(n)), 0.25, np.random)
        end = time.time()
        times_trials.append(end-start)
        num_queries = np.sum(queried)/2
        mistakes = count_mistakes(clusters, gold_mat)
        queries_trials.append(num_queries)
        mistakes_trials.append(mistakes)
        print('Trial ' + str(trial) + ' finished')

    print('Random query pivot')
    print('Mistakes: ' + str(np.mean(mistakes_trials)))
    print('Queries: ' + str(np.mean(queries_trials)))
    print('Avg. Time: ' + str(np.mean(times_trials)))

    if dataset in {'gym', 'landmarks', 'allsports'} and int(sys.argv[2]) in {0, 2}:
        print('LP Rounding')
        mistakes_trials = []
        times_trials = []
        for i in range(3):
            mistakes = 0
            start = time.time()
            clusters = lp_rounding(adj_mat, np.random, ilp=False)
            end = time.time()
            mistakes = count_mistakes(clusters, gold_mat)
            mistakes_trials.append(mistakes)
            times_trials.append(end-start)
            print('Trial ' + str(i) + ' Finished')
        mistakes = np.mean(mistakes_trials)
        print('Mistakes: ' + str(mistakes))
        print('Avg. Time: ' + str(np.mean(times_trials)))

        print('LP Rounding on Weighted Graph')
        mistakes_trials = []
        times_trials = []
        for i in range(3):
            mistakes = 0
            start = time.time()
            clusters = lp_rounding_weighted(adj_mat, np.random, ilp=False)
            end = time.time()
            if int(sys.argv[2]) == 2:
                mistakes = count_mistakes(clusters, weighted_adj_mat, weighted=True)
            else:
                mistakes = count_mistakes(clusters, gold_mat)
            mistakes_trials.append(mistakes)
            times_trials.append(end-start)
            print('Trial ' + str(i) + ' Finished')
            print(mistakes_trials)
            sys.stdout.flush()
        mistakes = np.mean(mistakes_trials)
        print('Mistakes: ' + str(mistakes))
        print('Avg. Time: ' + str(np.mean(times_trials)))
