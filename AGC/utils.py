import numpy as np
import robust_laplacian
import scipy
from typing import Tuple
from digeo import Mesh


def compute_mesh_laplacian(
    verts: np.ndarray, faces: np.ndarray
) -> Tuple[scipy.sparse.csc_matrix, np.ndarray]:
    lapl, mass = robust_laplacian.mesh_laplacian(verts, faces)
    return lapl, mass.diagonal()


def compute_eig_laplacian(
    lapl: scipy.sparse.csc_matrix,
    massvec: np.ndarray,
    k_eig: int = 128,
    eps: float = 1e-8,
) -> Tuple[np.typing.NDArray, np.typing.NDArray]:
    """Compute the eigendecomposition of the Laplacian

    Args:
        lapl: [N x N] Laplacian
        massvec: [N] mass vector
        k_eig (int, optional): number of eigenvalues and eigenvectors desired.
            Defaults to 10.
        eps (float, optional): constant used to perturb Laplacian during
            eigendecomposition. Defaults to 1e-8.

    Raises:
        ValueError: although multiple attempts were made, the eigendecomposition
            failed.

    Returns:
        Tuple[np.ndarray, np.ndarray]: k eigenvalues, [k x N] eigenvectors.
    """

    # Prepare matrices for eigendecomposition like in DiffusionNet code
    lapl_eigsh = (lapl + scipy.sparse.identity(lapl.shape[0]) * eps).tocsc()
    mass_mat = scipy.sparse.diags(massvec)
    eigs_sigma = eps

    failcount = 0
    while True:
        try:
            evals, evecs = scipy.sparse.linalg.eigsh(
                lapl_eigsh.astype(np.float32),
                k=k_eig,
                M=mass_mat.astype(np.float32),
                sigma=eigs_sigma,
            )
            evals = np.clip(evals, a_min=0.0, a_max=float("inf"))
            break
        except RuntimeError as exc:
            if failcount > 3:
                raise ValueError("failed to compute eigendecomp") from exc
            failcount += 1
            print("--- decomp failed; adding eps ===> count: " + str(failcount))
            lapl_eigsh = lapl_eigsh + scipy.sparse.identity(lapl.shape[0]) * (
                eps * 10**failcount
            )
    return evals, evecs


def heat_kernel_signature(mesh: Mesh, n_eig=128, n_scales=16, times=None):
    """
    Compute HKS for a given mesh.
    """
    lapl, massvec = compute_mesh_laplacian(
        mesh.vertices.cpu().numpy(), mesh.faces.cpu().numpy()
    )
    evals, evecs = compute_eig_laplacian(lapl, massvec, k_eig=n_eig, eps=10e-5)

    order = np.argsort(evals)
    evals = np.real(evals[order])
    evecs = np.real(evecs[:, order])

    # Sort eigenvalues/vectors ascending
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Choose time samples
    if times is None:
        times = np.logspace(-2, 0.0, num=n_scales)
    else:
        times = np.asarray(times, dtype=float)

    # Drop zero mode
    phi = evecs[:, 1:]
    lam = evals[1:]

    # Compute HKS
    phi2 = phi**2
    E = np.exp(np.outer(-lam, times))
    H = phi2 @ E

    return H
