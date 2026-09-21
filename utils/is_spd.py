import torch

def is_spd(P, rtol=1e-5, atol=1e-8):
    """
    Vérifie si une matrice est Symétrique Définie Positive (SPD).
    """
    P_T = P.transpose(-1, -2)
    if not torch.allclose(P, P_T, rtol=rtol, atol=atol):
        return False

    try:
        torch.linalg.cholesky(P)
        return True
    except RuntimeError:
        return False