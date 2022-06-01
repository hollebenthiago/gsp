import pygsp as p

import numpy as np
from numpy.linalg import matrix_rank, inv, pinv

from scipy.optimize import linprog, minimize, Bounds

from math import cos, asin, sqrt, pi

def vector_to_matrix(n, ws):
    
    W = np.zeros((n, n))
    indexes = np.triu_indices(n, k = 1)
    W[indexes] = ws
    W[indexes[::-1]] = ws
    degrees = [sum(W[i]) for i in range(n)]
    
    return W, degrees

def objective_function_spielman(x, ws, verbose):
    
    n, d = np.shape(x)
    W, ds = vector_to_matrix(np.shape(x)[0], ws)
    L = np.diag(ds) - W
    if verbose:
        print(np.linalg.norm(L @ x))
    return np.linalg.norm(L @ x)

def constraints_spielman(n, c, α, positive):
    
    _, ds = vector_to_matrix(n, c)
    if positive:
        g = lambda c: α * n - sum([max(0, 1 - di) ** 2 for di in ds]) 
    else:
        g = lambda c: sum([max(0, 1 - di) for di in ds]) - α * n 
    return ({'type': 'ineq', 'fun': g})

def bounds_spielman(n):
    
    return n * [(0, 1)]

def make_graph(n, ws, tol = 1e-10):
    
    ws = ws.copy()
    ws[np.abs(ws) < tol] = 0
    new_ws = ws
    W = np.zeros((n, n))
    W[np.triu_indices(n, 1)] = new_ws
    W += W.T

    return p.graphs.Graph(W)

def great_circle_distance(lat1, lon1, lat2, lon2):
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 12742 * asin(sqrt(a)) #2*R*asin...

