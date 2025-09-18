import time
import scipy as sp
import numpy as np
import networkx as nx
from numba import jit, njit, prange, uint32, uint64, float64, types, jit_module
from numba.typed import List, Dict

UINT32_MAX = np.iinfo(np.uint32).max

class Poset:
    def __init__(self, G, verbose=False, use_nx_matching=True):
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("G must be a directed acyclic graph")
        self.G = G
        self.V = G.number_of_nodes()
        self.use_nx_matching = use_nx_matching
        self.verbose = verbose

        SC = nx.to_scipy_sparse_array(G,nodelist=list(range(self.V)),dtype=np.intc,format='csr')
        self.row = SC.indptr
        self.col = SC.indices

        # run decomposition
        if self.verbose:
            print("Poset initialization")
            tic = time.time()
        self.chain_id, self.successor, self.n_chains = self.__chain_decomposition()
        if self.verbose:
            toc = time.time()
            print(f"Chain decomposition took {toc - tic:.2f} seconds")
            tic = time.time()

        self.nivcol, self.nivrow = self.__make_niv_csr_compressed()
        if self.verbose:
            toc = time.time()
            print(f"NIV construction took {toc - tic:.2f} seconds")
            tic = time.time()

        self.levelcol, self.levelrow = self.__prep_parallel()
        if self.verbose:
            toc = time.time()
            print(f"Parallel prep took {toc - tic:.2f} seconds")
        
    def __maximum_matching(self):
        mt = np.ones(self.V, dtype=np.int32) * -1
        used = [False for _ in range(self.V)]
        entered = [False for _ in range(self.V)]
        parent = [-1 for _ in range(self.V)]
        current = [self.row[i] for i in range(self.V)]
        end = [self.row[i + 1] for i in range(self.V)]


        def kuhn(v):
            # iterative Kuhn's algorithm.
            stack = [v]

            while stack:
                v = stack[-1]

                # If we are entering this node for the first time
                if not entered[v]:
                    if used[v]:
                        stack.pop()
                        continue
                    used[v] = True
                    entered[v] = True

                # If we have explored all edges, backtrack
                if current[v] >= end[v]:
                    stack.pop()
                    continue

                to = self.col[current[v]]
                current[v] += 1
                mate = mt[to]

                if mate == -1:
                    mt[to] = v
                    while stack:
                        cur = stack.pop()
                        parent_to = parent[cur]
                        if parent_to is None:
                            break
                        mt[parent_to] = v
                    return True

                if not used[mate]:
                    stack.append(mate)
                    parent[mate] = to

            return False

        for v in range(self.V):
            for j in range(self.row[v], self.row[v + 1]):
                to = self.col[j]
                if mt[to] == -1:
                    mt[to] = v
                    used[v] = True
                    break

        for v in range(self.V):
            if not used[v]:
                used[v] = True
                kuhn(v)

        return mt

    def __make_successor(self):
        successor = {i: i for i in range(self.V)}
        if self.use_nx_matching:
            G = self.G
            nodes = list(G.nodes())
            left = [(0, node) for node in nodes]
            right = [(1, node) for node in nodes]

            B = nx.Graph()
            B.add_nodes_from(left, bipartite=0)
            B.add_nodes_from(right, bipartite=1)
            B.add_edges_from(((0, u), (1, v)) for u, v in G.edges())
                
            matching = nx.algorithms.bipartite.maximum_matching(B, top_nodes=left)

            for node in nodes:
                mate = matching.get((0, node))
                if mate is not None:
                    successor[node] = mate[1]

        else:
            mt = self.__maximum_matching()

            for i in range(self.V):
                successor[i] = i
            
            for i in range(self.V):
                if mt[i] != -1:
                    successor[mt[i]] = i

        return successor

    def __chain_decomposition(self):
        """
        Return (chain_id, successor, n_chains) for the poset whose Hasse diagram is G.
        `chain_id[v]` gives the chain index for node v.
        `successor[v]` is the next node in that chain (or v itself at the chain tail).
        """
        successor = self.__make_successor()

        chain_id: Dict[Hashable, int] = {}
        n_chains = 0
        for start in range(self.V):
            if start in chain_id:
                continue
            current = start
            while current not in chain_id:
                chain_id[current] = n_chains
                nxt = successor[current]
                if nxt == current or nxt in chain_id:
                    break
                current = nxt
            n_chains += 1

        np_chain = np.empty(self.V, dtype=np.int32)
        np_succ = np.empty(self.V, dtype=np.int32)
        for node in range(self.V):
            np_chain[node] = chain_id[node]
            np_succ[node] = successor[node]

        np_chain = np.asarray(np_chain, dtype=np.int32, order='C')
        np_succ = np.asarray(np_succ, dtype=np.int32, order='C')

        return np_chain, np_succ, n_chains

    @staticmethod
    @njit
    def __jitted_niv_csr_compressed(V, chain_id, successor, row, col, n_chains):
        nivcol: list[int] = []
        nivrow = [0, 0]
        nivfield = np.full(n_chains, UINT32_MAX, dtype=np.uint32)
        inserted: list[int] = []

        for v in range(V - 1, -1, -1):
            v_id = chain_id[v]
            for j in range(row[v], row[v + 1]):
                w = col[j]
                if w < nivfield[chain_id[w]]:

                    if chain_id[w] != v_id:
                        if nivfield[chain_id[w]] == UINT32_MAX:
                            inserted.append(chain_id[w])
                        nivfield[chain_id[w]] = w

                    for k in range(nivrow[V - w], nivrow[V - w + 1]):
                        p = nivcol[k]
                        if chain_id[p] == v_id:
                            continue
                        if p < nivfield[chain_id[p]]:
                            if nivfield[chain_id[p]] == UINT32_MAX:
                                inserted.append(chain_id[p])
                            nivfield[chain_id[p]] = p

            while len(inserted) > 0:
                u = inserted.pop()
                nivcol.append(nivfield[u])
                nivfield[u] = UINT32_MAX
            nivrow.append(len(nivcol))


        nivcol = np.asarray(nivcol, dtype=np.uint32)
        nivrow = np.asarray(nivrow, dtype=np.uint64)
        return nivcol, nivrow

    def __make_niv_csr_compressed(self):
        # get graph as sparse matrix
        row = self.row
        col = self.col

        return self.__jitted_niv_csr_compressed(self.V, self.chain_id, self.successor, row, col, self.n_chains)
        

    def __prep_parallel(self):
        row = self.row
        col = self.col

        revcol = np.zeros_like(col)
        revrow = np.zeros_like(row)
        leaves = []
        outdegrees = np.zeros(self.V, dtype=np.int32)
        levelcol = np.zeros(self.V, dtype=np.int32)
        
        
        # prepare all the structures
        g = [[] for _ in range(self.V)]
        for i in range(self.V):
            for j in range(row[i], row[i + 1]):
                g[col[j]].append(i)

            outdegrees[i] = row[i + 1] - row[i]
            if row[i + 1] - row[i] == 0:
                leaves.append(i)

        inserted = 0
        revrow[0] = 0
        for i in range(self.V):
            for j in g[i]:
                revcol[inserted] = j
                inserted += 1
            revrow[i + 1] = inserted


        # compute antichains
        index = 0
        while len(leaves) > 0:
            l = leaves.pop()
            levelcol[index] = l
            index += 1
        
        levelrow = []

        levelrow.append(0)
        levelrow.append(index)

        remaining = self.V - index
        i = 0
        j = 0
        while remaining > 0:
            for k in range(revrow[levelcol[j]], revrow[levelcol[j] + 1]):
                outdegrees[revcol[k]] -= 1
                if outdegrees[revcol[k]] == 0:
                    levelcol[index] = revcol[k]
                    index += 1
                    remaining -= 1
            
            if j == levelrow[i + 1] - 1:
                levelrow.append(index)
                i += 1
            j += 1
            
        levelrow.append(index)

        levelrow = np.asarray(levelrow, dtype=np.uint64)
        levelcol = np.asarray(levelcol, dtype=np.uint32)

        return levelcol, levelrow

    def zeta(self, x, parallel=False):
        if parallel:
            return self._zeta_parallel(x, self.successor, self.nivcol, self.nivrow, self.V, self.levelcol, self.levelrow)
        else:
            return self._zeta(x, self.successor, self.nivcol, self.nivrow, self.V)

    @staticmethod
    @njit
    def _zeta(x, successor, nivcol, nivrow, V):
        y = np.zeros_like(x)
        cache = np.zeros(V)
        for i in range(V - 1, -1, -1):
            cache[i] = x[i] + cache[successor[i]]
            acc = cache[i]
            for j in range(nivrow[V - i], nivrow[V - i + 1]):
                acc += cache[nivcol[j]]
            y[i] = acc
        return y

    @staticmethod
    @njit(parallel=True)
    def _zeta_parallel(x, successor, nivcol, nivrow, V, levelcol, levelrow):
        y = np.zeros_like(x)
        cache = np.zeros(V)
        cur_level = 0
        while cur_level < len(levelrow) - 1:
            for mine in prange(levelrow[cur_level], levelrow[cur_level + 1]):
                i = levelcol[mine]
                cache[i] = x[i] + cache[successor[i]]
                acc = cache[i]
                for j in range(nivrow[V - i], nivrow[V - i + 1]):
                    acc += cache[nivcol[j]]
                y[i] = acc
            cur_level += 1
        return y

    def mobius(self, x, parallel=False):
        if parallel:
            return self._mobius_parallel(x, self.successor, self.nivcol, self.nivrow, self.V, self.levelcol, self.levelrow)
        else:
            return self._mobius(x, self.successor, self.nivcol, self.nivrow, self.V)

    @staticmethod
    @njit
    def _mobius(x, successor, nivcol, nivrow, V):
        y = np.zeros_like(x)
        cache = np.zeros(V)
        for i in range(V - 1, -1, -1):
            y[i] = 0
            cache[i] = cache[successor[i]]
            acc = x[i] - cache[successor[i]]
            for j in range(nivrow[V - i], nivrow[V - i + 1]):
                acc -= cache[nivcol[j]]
            y[i] = acc
            cache[i] += y[i]
        return y

    @staticmethod
    @njit(parallel=True)
    def _mobius_parallel(x, successor, nivcol, nivrow, V, levelcol, levelrow):
        y = np.zeros_like(x)
        cache = np.zeros(V)
        cur_level = 0
        while cur_level < len(levelrow) - 1:
            for mine in prange(levelrow[cur_level], levelrow[cur_level + 1]):
                i = levelcol[mine]
                y[i] = 0
                cache[i] = cache[successor[i]]
                acc = x[i] - cache[successor[i]]
                for j in range(nivrow[V - i], nivrow[V - i + 1]):
                    acc -= cache[nivcol[j]]
                y[i] = acc
                cache[i] += y[i]
            cur_level += 1
        return y

jit_module(nopython=True, error_model="numpy")
