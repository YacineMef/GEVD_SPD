import torch

def is_symmetric(P, rtol=1e-5, atol=1e-8):
    """
    Vérifie si une matrice est symétrique.
    """
    P_T = P.transpose(-1, -2)
    return torch.allclose(P, P_T, rtol=rtol, atol=atol)