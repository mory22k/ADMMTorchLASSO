import torch
import torch.nn.functional as F
from tqdm import tqdm


def soft_thresholding_vec(
    x_vec: torch.Tensor, threshold_vec: torch.Tensor
) -> torch.Tensor:
    """Applies soft-thresholding to each element of a vector.

    Args:
        x_vec (torch.Tensor): Input tensor.
        threshold_vec (torch.Tensor): Threshold tensor.

    Returns:
        torch.Tensor: Soft-thresholded output.
    """
    return torch.sign(x_vec) * F.relu(x_vec.abs() - threshold_vec)


def admm(
    x_mat: torch.Tensor,
    y_vec: torch.Tensor,
    lam_vec: torch.Tensor,
    z_init_vec: torch.Tensor,
    u_init_vec: torch.Tensor,
    rho_vec: torch.Tensor,
    max_iter: int = 100,
    torr: float = 1e-4,
    return_history: bool = False,
    show_progress_bar: bool = True,
    verbose: bool = False,
):
    """Performs the ADMM optimization procedure.

    Args:
        x_mat (torch.Tensor): Input feature matrix (n_samples, n_features).
        y_vec (torch.Tensor): Target vector (n_samples,).
        lam_vec (torch.Tensor): Regularization parameter vector (n_features,).
        z_init_vec (torch.Tensor): Initial z vector (n_features,).
        u_init_vec (torch.Tensor): Initial u vector (n_features,).
        rho_vec (torch.Tensor): Rho parameter vector (n_features,).
        max_iter (int): Maximum iterations. Default is 100.
        torr (float): Tolerance for convergence. Default is 1e-4.
        return_history (bool): Whether to return history of variables. Default is False.
        show_progress_bar (bool): Whether to show tqdm progress bar. Default is True.
        verbose (bool): Whether to print progress. Default is False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list, list, list]:
        - Final w, z, u vectors
        - (Optional) History of w, z, u if return_history=True.
    """
    num_data_points, num_variables = x_mat.shape

    rho_diag = torch.diag(rho_vec)
    a_mat = x_mat.t().matmul(x_mat) + rho_diag
    a_mat_chol = torch.linalg.cholesky(a_mat)
    threshold_vec = lam_vec / rho_vec

    w_vec = torch.zeros(num_variables, dtype=x_mat.dtype, device=x_mat.device)
    z_vec = z_init_vec.clone()
    u_vec = u_init_vec.clone()

    w_vec_history, z_vec_history, u_vec_history = [], [], []

    iterator = range(max_iter)
    if show_progress_bar:
        iterator = tqdm(iterator, total=max_iter, desc="ADMM")

    for iter in iterator:
        b = x_mat.t().matmul(y_vec) + rho_diag.matmul(z_vec - u_vec)
        w_vec = torch.cholesky_solve(b.unsqueeze(1), a_mat_chol, upper=False).squeeze(1)

        w_plus_u = w_vec + u_vec
        z_vec = soft_thresholding_vec(w_plus_u, threshold_vec)
        u_vec = u_vec + (w_vec - z_vec)

        if return_history:
            w_vec_history.append(w_vec.detach().clone())
            z_vec_history.append(z_vec.detach().clone())
            u_vec_history.append(u_vec.detach().clone())

        if iter % 10 == 0:
            if verbose:
                norm = torch.norm(w_vec - z_vec)
                tqdm.write(f"iter: {iter}, norm: {norm}")
        if torch.norm(w_vec - z_vec) < torr:
            if show_progress_bar:
                tqdm.write(f"ADMM converged at iteration {iter}")
            break

    if return_history:
        return (
            torch.stack(w_vec_history),
            torch.stack(z_vec_history),
            torch.stack(u_vec_history),
        )
    return w_vec, z_vec, u_vec
