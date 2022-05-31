from ast import Lambda
import pygsp as p

import numpy as np
from numpy.linalg import matrix_rank, inv, pinv
import matplotlib.pyplot as plt

import cv2

from scipy import fft

from scipy.sparse.linalg import eigs
from scipy.linalg import eig



def start_graph(reference_operator, G):
    
    if reference_operator == 'laplacian':
        G.compute_fourier_basis()
        V = G.U
        Λ = np.diag(G.e)

    if reference_operator == 'adjacency':
        λs, vs = np.linalg.eig(G.A.todense())
        V = vs
        Λ = np.diag(λs)

    G.Λ = Λ/np.max(Λ)
    G.V = np.array(V)/np.max(Λ)

def optimal_sampling_operator(V, k, m, sort = True, verbose = False):
    Vk = V[:, :k]
    M = []
    n = np.shape(Vk)[0]
    check_list = list(np.arange(0, n, 1, dtype = int))
    while len(M) < m:
        c = 0
        i = 0
        for j in check_list:
            if j in M:
                continue
            d = min(np.linalg.svd(Vk[M + [i]])[1])
            if d >= c:
                i = j
                c = d
        check_list.remove(i)
        M += [i]
        if verbose:
            print(len(check_list))
    if sort:
        M.sort()
    return M    

def setup(reference_operator, G, indexes):
    '''
    reference_operator = laplacian or adjacency
    G the graph
    indexes the indexes of the samples
    it is assumed that the number of collumns given is equal to the length of indexes
    ψ sampling matrix
    U inverse of ψVk
    φ interpolation matrix
    
    return: V, Λ, ψ, U, φ
    '''
    
    start_graph(reference_operator, G)
        
    m = np.shape(G.V)[0]
    k = len(indexes) #number of samples
    ψ = np.zeros((k, np.shape(G.V)[0]))
    for i, o_i in enumerate(indexes):
        ψ[i][o_i] = 1
    U = inv(ψ @ G.V[:, :k])
    φ = G.V[:, :k] @ U
    return G.V, G.Λ, ψ, U, φ

def sampling_matrix(V, indexes):
    ψ = np.zeros((np.shape(V)[0], np.shape(V)[0]))
    for i in indexes:
        ψ[i][i] = 1
    return ψ

def make_signal(rs, G, k):
    coefs = rs.uniform(100 * np.ones(k))
    signal = np.array(G.V[:, :k] @ coefs).reshape(G.N,)
    return signal