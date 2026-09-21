# GEVD

import torch

# gved
from collections.abc import Callable
from torch.autograd import Function
from yetanotherspdnet.functions.spd_linalg import (
    CongruenceRectangular,
    congruence_rectangular,
    symmetrize,
    _aux_eigh_operation_grad,
)
from yetanotherspdnet.functions.scalar_functions import inv, sqrt_derivative



def gevd(
    P: torch.Tensor, Q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the generalized eigenvalue decomposition of an SPD matrix P and a symmetric matrix Q

    We have :math: `P = A^\top A, Q = A^\top Lambda A`.

    Parameters
    ----------
    P : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    Q : torch.Tensor of shape (n_features, n_features)
        Symmetric matrix

    Returns
    -------
    Lambda : torch.Tensor of shape (n_features,)
        Generalized eigenvalues

    A : torch.Tensor of shape (n_features, n_features)
        Generalized eigenvectors

    B : torch.Tensor of shape (n_features, n_features)
        Inverse of A
    """
    L = torch.linalg.cholesky(P)
    L_inv = torch.inverse(L)
    Lambda, U = torch.linalg.eigh(L_inv @ Q @ L_inv.transpose(-2, -1))
    return Lambda, (L @ U).transpose(-2, -1), L_inv.transpose(-2, -1) @ U

def gevd_operation(
    Lambda: torch.Tensor, A: torch.Tensor, operation: Callable
) -> torch.Tensor:
    """
    Applies a function on the generalized eigenvalues of the generalized eigenvalue decomposition
    of a symmetric matrix Q and a SPD matrix P.

    We have :math: `P = A^\top A, Q = A^\top Lambda A`.

    It returns :math: `A^\top f(Lambda) A`.

    Parameters
    ----------
    Lambda : torch.Tensor of shape (n_features,)
        Generalized eigenvalues

    A : torch.Tensor of shape (n_features, n_features)
        Generalized eigenvectors

    operation : Callable
        Function to apply on generalized eigenvalues

    Returns
    -------
    result : torch.Tensor of shape (n_features, n_features)
        Resulting symmetric matrix with operation applied to generalized eigenvalues
    """
    Lambda, A = (
        torch.real(Lambda),
        torch.real(A),
    )  # to correct eventual numerical errors...
    _Lambda = operation(Lambda)
    result = (A.transpose(-1, -2) * _Lambda.unsqueeze(-2)) @ A
    return result


def gevd_operation_grad(
    grad_output: torch.Tensor,
    Lambda: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    t: float | torch.Tensor,
    operation: Callable,
    operation_derivative: Callable,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the backpropagation of the gradient for a function applied on the
    generalized eigenvalues of the generalized eigenvalue decomposition of a
    symmetric matrix Q and a SPD matrix P.

    We have :math: `P = A^\top A, Q = A^\top Lambda A, A^{-1} = B`.

    Parameters
    ----------
    grad_output : torch.Tensor of shape (n_features, n_features)
        Gradient of the loss with respect to the output of the operation on generalized eigenvalues

    Lambda : torch.Tensor of shape (n_features,)
        Generalized eigenvalues

    A : torch.Tensor of shape (n_features, n_features)
        Generalized eigenvectors

    B : torch.Tensor of shape (n_features, n_features)
        Inverse of A

    t : float | torch.Tensor
        parameter on the path, should be in [0,1]

    operation : Callable
        Function to apply on generalized eigenvalues

    operation_derivative : Callable
        Derivative of the function to apply on generalized eigenvalues

    Returns
    -------
    grad_P : torch.Tensor of shape (n_features, n_features)
        Gradient w.r.t. SPD matrix P

    grad_Q : torch.Tensor of shape (n_features, n_features)
        Gradient w.r.t. symmetric matrix Q

    grad_t : torch.Tensor (implementation of my own)
        Gradient w.r.t. t
    """
    grad_out_cong = A @ symmetrize(grad_output) @ A.transpose(-1, -2)

    # gradient w.r.t. (own implementation) 
    Lambda_t = torch.pow(Lambda, t)
    log_Lambda = torch.log(Lambda)
    diag_grad_out_cong = torch.diagonal(grad_out_cong, dim1=-2, dim2=-1)
    grad_t = (diag_grad_out_cong * Lambda_t * log_Lambda).sum(-1)

    # gradient w.r.Q
    aux_mat_Q = _aux_eigh_operation_grad(Lambda, operation, operation_derivative)
    middle_term_Q = aux_mat_Q * grad_out_cong
    grad_Q = B @ middle_term_Q @ B.transpose(-1, -2)
    
    # gradient w.r.P
    _Lambda = operation(Lambda)
    middle_term_P = symmetrize(
        grad_out_cong * _Lambda.unsqueeze(-2) - middle_term_Q * Lambda.unsqueeze(-2)
    )
    grad_P = B @ middle_term_P @ B.transpose(-1, -2)
    
    return grad_P, grad_Q, grad_t


# ----------------------------------------
# Affine-invariant geodesics based on GEVD
# ----------------------------------------
def affine_invariant_geodesic_gevd(
    P: torch.Tensor, Q: torch.Tensor, t: float | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Affine-invariant geodesic based on generalized eigenvalue decomposition of P and Q.

    Parameters
    ----------
    P : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    Q : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    t : float | torch.Tensor
        parameter on the path, should be in [0,1]

    Returns
    -------
    S : torch.Tensor of shape (n_features, n_features)
        SPD matrix

    Lambda : torch.Tensor of shape (n_features,)
        Generalized eigenvalues of Q w.r.t. P

    A : torch.Tensor of shape (n_features, n_features)
        Generalized eigenvectors

    B : torch.Tensor of shape (n_features, n_features)
        Inverse of A
    """
    Lambda, A, B = gevd(P, Q)
    pow_t = lambda x: torch.pow(x, t)
    return gevd_operation(Lambda, A, pow_t), Lambda, A, B


class AffineInvariantGeodesicGEVD(Function):
    """
    Affine-invariant geodesic based on generalized eigenvalue decomposition
    """

    @staticmethod
    def forward(
        ctx, P: torch.Tensor, Q: torch.Tensor, t: float | torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the affine-invariant geodesic based on generalized eigenvalue decomposition of P and Q.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        P : torch.Tensor of shape (n_features, n_features)
            SPD matrix

        Q : torch.Tensor of shape (n_features, n_features)
            SPD matrix

        t : float | torch.Tensor
            parameter on the path, should be in [0,1]

        Returns
        -------
        S : torch.Tensor of shape (n_features, n_features)
            SPD matrix
        """
        S, Lambda, A, B = affine_invariant_geodesic_gevd(P, Q, t)
        ctx.t = t
        ctx.save_for_backward(Lambda, A, B)
        return S

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Backward pass of the affine-invariant geodesic based on generalized eigenvalue decomposition of P and Q.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass

        grad_output : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss w.r.t. the output of the affine-invariant geodesic between P and Q

        Returns
        -------
        grad_P : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss w.r.t. P

        grad_Q : torch.Tensor of shape (n_features, n_features)
            Gradient of the loss w.r.t. Q

        grad_t : torch.Tensor
            Gradient of the loss w.r.t. t
        """
        t = ctx.t
        Lambda, A, B = ctx.saved_tensors
        pow_t = lambda x: torch.pow(x, t)
        pow_t_deriv = lambda x: t * torch.pow(x, t - 1)
        grad_P, grad_Q, grad_t = gevd_operation_grad(
            grad_output, Lambda, A, B, t, pow_t, pow_t_deriv
        )
        # Scaling the gradient of t 
        # step_norm = torch.log(Lambda).norm(dim=-1).clamp_min(1e-6)
        # grad_t = grad_t / step_norm
        
        return grad_P, grad_Q, grad_t