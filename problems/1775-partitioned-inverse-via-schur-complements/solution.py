import numpy as np

def _blocks(M, k):
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    if M.ndim != 2 or M.shape[1] != n:
        raise ValueError("M must be square")
    if not 1 <= k < n:
        raise ValueError("need 1 <= k < n")
    return M[:k, :k], M[:k, k:], M[k:, :k], M[k:, k:]

def partitioned_inverse(M, k):
    E, F, G, H =_blocks(M, k)
    Hinv_G = np.linalg.solve(H, G)
    S = E - F @ Hinv_G
    S_inv = np.linalg.inv(S)
    H_inv = np.linalg.inv(H)
    F_Hinv = np.linalg.solve(H.T, F.T).T

    top_left = S_inv
    top_right = -S_inv @ F_Hinv
    bottom_left = -Hinv_G @ S_inv
    bottom_right = H_inv + Hinv_G @ S_inv @ F_Hinv

    return np.block([[top_left, top_right], [bottom_left, bottom_right]])

def block_slogdet(M, k):
    E, F, G, H = _blocks(M, k)
    S = E - F @ np.linalg.solve(H, G)
    sing_S , logdet_S = np.linalg.slogdet(S)
    sing_H , logdet_H = np.linalg.slogdet(H)
    return sing_S * sing_H, logdet_H + logdet_S

def bordered_inverse(A_inv, b, c):
    A_inv = np.asarray(A_inv, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = A_inv.shape[0]
    if A_inv.shape != (n, n) or b.shape != (n,):
        raise ValueError("shape mismatch between A_inv and b")

    u = A_inv @ b
    s = float(c) - b @ u
    if abs(s) < 1e-12:
        raise ValueError("Schur complement s = c - b^T A^{-1} b is (near) zero")

    out = np.empty((n + 1, n + 1))
    out[:n, :n] = A_inv + np.outer(u, u) / s 
    out[:n, n] = -u / s 
    out[n, :n] = -u / s 
    out[n, n] = 1.0 / s 
    return out
