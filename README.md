# Fast Möbius and Zeta Transforms

Code to execute Möbius and Zeta Transforms given a poset and a signal.

## Dependencies

The code uses networkx, numpy and numba. All versions are given in the requirements.txt

```pip install -r requirements.txt```


## Usage

A demo is given in `demo.ipynb`. 
Minimal example:

```
G = nx.DiGraph(...)            # Poset defined as a networkx graph
P = Poset(G,                   # Input poset
    verbose=False,             # Add some timing prints
    use_nx_matching=False      # True: use networkx matching function (slower),
                               # False: use Kuhn matching algorithm (faster)
    )

x = np.random.rand(V)

y = P.zeta(x,   parallel=True) # parallel=True uses multithreading, useful for larger posets
z = P.mobius(x, parallel=True) # parallel=True uses multithreading, useful for larger posets


```

