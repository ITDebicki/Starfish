import torch
from typing import Union

def rbf_kernel(X: torch.Tensor, Z: torch.Tensor, variance: float, lengthscale: Union[torch.Tensor, float]) -> torch.Tensor:
    """
    A classic radial-basis function (Gaussian; exponential squared) covariance kernel

    .. math::
        \\kappa(X, Z | \\sigma^2, \\Lambda) = \\sigma^2 \\exp\\left[-\\frac12 (X-Z)^T \\Lambda^{-1} (X-Z) \\right]

    Parameters
    ----------
    X : torch.Tensor
        The first set of points
    Z : torch.Tensor
        The second set of points. Must have same second dimension as `X`
    variance : double
        The amplitude for the RBF kernel
    lengthscale : torch.Tensor or double
        The lengthscale for the RBF kernel. Must have same second dimension as `X`

    """
    sq_dist = torch.cdist(X / lengthscale, Z / lengthscale).pow(2)
    return variance * torch.exp(-0.5 * sq_dist)

def batch_kernel(X:torch.Tensor, Z:torch.Tensor, variances:torch.Tensor, lengthscales:torch.Tensor) -> torch.Tensor:
    """
    Batched RBF kernel

    Parameters
    ----------
    X : torch.Tensor
        The first set of points
    Z : torch.Tensor
        The second set of points. Must have same second dimension as `X`
    variances : torch.Tensor
        The amplitude for the RBF kernel
    lengthscales : torch.Tensor
        The lengthscale for the RBF kernel. Must have same second dimension as `X`

    See Also
    --------
    :function:`rbf_kernel`
    """

    blocks = [rbf_kernel(X, Z, var, ls) for var, ls in zip(variances, lengthscales)]
    return torch.block_diag(*blocks)