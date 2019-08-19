import sys
import math
import numpy as np
from copy import deepcopy
import time
from gurobipy import *
from algorithms import *
np.random.seed(1555)

if __name__ == '__main__':
    mode = sys.argv[1].upper()
    n = 0
    adj_mat = None
    sizes = None
    if mode == 'N': # Sizes are i.i.d. Gaussian
        num_cliques = 10
        sizes = np.random.normal(loc=8, scale=2, size=num_cliques)
        # num_cliques = 3
        # sizes = np.random.normal(loc=3, scale=1, size=num_cliques)
        sizes = [int(s) for s in sizes]
    elif mode == 'D': # Dirichlet distribution (1 large cluster, others small)
        num_cliques = 3
        alpha = [5]+[1]*(num_cliques-1)
        n = 100
        sizes = np.random.dirichlet(alpha, size=1)[0]*n
        sizes = [int(s) for s in sizes]
    elif mode == 'S': # S -- Fixed (non-random) skewed cluster sizes
        num_cliques = 10
        sizes = [5]*5+[15]*4+[30]
    n = sum(sizes)
    adj_mat = -1*np.ones((n, n), dtype=np.int32)
    for c in range(num_cliques):
        for i in range(sum(sizes[:c]), sum(sizes[:c+1])):
            for j in range(sum(sizes[:c]), sum(sizes[:c+1])):
                adj_mat[i][j] = 1
    edge_choice = int(sys.argv[2])
    L = int(sys.argv[3])
    oracle_mat = deepcopy(adj_mat)
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
        for c in range(num_cliques):
            count = 0
            flipped_set = set()
            while count < min(L/num_cliques, sum(sizes[:c+1])-sum(sizes[:c])-1):
                row = np.random.randint(sum(sizes[:c]), sum(sizes[:c+1]))
                col = np.random.randint(sum(sizes[:c]), sum(sizes[:c+1]))
                if row != col and (row, col) not in flipped_set and (col, row) not in flipped_set:
                    adj_mat[row][col] *= -1
                    adj_mat[col][row] *= -1
                    count += 1
                    flipped_set.add((row, col))
            num_flipped_edges += count
        L = num_flipped_edges
    elif edge_choice == 3:
        num_flipped_edges = 0
        for c in range(num_cliques):
            count = 0
            flipped_set = set()
            while count < min(L/num_cliques, sum(sizes[:c+1])-sum(sizes[:c])-1):
                row = np.random.randint(sum(sizes[:c]), sum(sizes[:c+1]))
                col = np.random.randint(sum(sizes[:c]), sum(sizes[:c+1]))
                if row != col and (row, col) not in flipped_set and (col, row) not in flipped_set:
                    adj_mat[row][col] *= -1
                    adj_mat[col][row] *= -1
                    count += 1
                    flipped_set.add((row, col))
            num_flipped_edges += count
        L = num_flipped_edges
        num_flipped_edges = 0
        for c in range(num_cliques):
            for c2 in range(c+1, num_cliques):
                count = 0
                num_to_flip = int(math.ceil(0.01*sizes[c]*sizes[c2]))
                flipped_set = set()
                while count < num_to_flip:
                    row = np.random.randint(sum(sizes[:c]), sum(sizes[:c+1]))
                    col = np.random.randint(sum(sizes[:c2]), sum(sizes[:c2+1]))
                    if (row, col) not in flipped_set and (col, row) not in flipped_set:
                        adj_mat[row][col] *= -1
                        adj_mat[col][row] *= -1
                        count += 1
                        flipped_set.add((row, col))
                num_flipped_edges += count
        L += num_flipped_edges

    print('ILP Oracle')
    start = time.time()
    clusters = lp_rounding(adj_mat, np.random, ilp=True)
    end = time.time()
    oracle_mat = -1*np.ones((n, n), dtype=np.int32)
    for clust in clusters:
        for i in clust:
            for j in clust:
                oracle_mat[i][j] = 1
    mistakes = count_mistakes(clusters, adj_mat)
    print('Mistakes: ' + str(mistakes))
    print('Time: ' + str(end-start))

    start = time.time()
    clusters = cautious(adj_mat, set(range(n)))
    end = time.time()
    mistakes = count_mistakes(clusters, adj_mat)
    print('Cautious')
    print('Mistakes: ' + str(mistakes))
    print('Time: ' + str(end-start))

    start = time.time()
    clusters, queries = bocker_alg(adj_mat, oracle_mat)
    end = time.time()
    mistakes = count_mistakes(clusters, adj_mat)
    print('Bocker 1')
    print('Mistakes: ' + str(mistakes))
    print('Queries: ' + str(queries))
    print('Time: ' + str(end-start))

    mistakes_trials = []
    times_trials = []
    print('3-approximation Pivot')
    for trial in range(3):
        start = time.time()
        clusters = three_approx(adj_mat, set(range(n)), np.random)
        end = time.time()
        times_trials.append(end-start)
        mistakes = count_mistakes(clusters, adj_mat)
        mistakes_trials.append(mistakes)
        print('Trial ' + str(trial) + ' finished')
    print('L: ' + str(L))
    print('Mistakes: ' + str(np.mean(mistakes_trials)))
    print('Time: ' + str(np.mean(times_trials)))

    print('Query pivot')
    start = time.time()
    clusters, queried = query_pivot(adj_mat, oracle_mat, set(range(n)))
    end = time.time()
    alg_time = end-start
    num_queries = np.sum(queried)/2
    mistakes = count_mistakes(clusters, adj_mat)
    print('L: ' + str(L))
    print('Mistakes: ' + str(mistakes))
    print('Queries: ' + str(num_queries))
    print('Time: ' + str(alg_time))

    queries_trials = []
    mistakes_trials = []
    times_trials = []
    print('Random query pivot')
    for _ in range(3):
        start = time.time()
        clusters, queried = random_query_pivot(adj_mat, oracle_mat, set(range(n)), 0.25, np.random)
        end = time.time()
        times_trials.append(end-start)
        num_queries = np.sum(queried)/2
        mistakes = count_mistakes(clusters, adj_mat)
        queries_trials.append(num_queries)
        mistakes_trials.append(mistakes)
        print('Trial ' + str(trial) + ' finished')
    print('Mistakes: ' + str(np.mean(mistakes_trials)))
    print('Queries: ' + str(np.mean(queries_trials)))
    print('Time: ' + str(np.mean(times_trials)))

    print('LP Rounding')
    mistakes_trials = []
    times_trials = []
    for _ in range(3):
        mistakes = 0
        start = time.time()
        clusters = lp_rounding(adj_mat, np.random, ilp=False)
        end = time.time()
        mistakes = count_mistakes(clusters, adj_mat)
        mistakes_trials.append(mistakes)
        times_trials.append(end-start)
        print('Trial ' + str(trial) + ' finished')
    mistakes = np.mean(mistakes_trials)
    print('Mistakes: ' + str(mistakes))
    print('Time: ' + str(np.mean(times_trials)))
