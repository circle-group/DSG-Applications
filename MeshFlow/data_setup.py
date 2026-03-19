from pathlib import Path
import os
import igl
import numpy as np
import torch
import scipy

# Code from "Flow Matching on General Geometries", Ricky T. Q. et al. (https://github.com/facebookresearch/riemannian-fm)
# CC BY-NC 4.0 License.


def get_eigfn(v, f, numeigs: int):
    assert numeigs <= v.shape[0], (
        "Cannot compute more eigenvalues than the number of vertices."
    )

    M = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
    L = -igl.cotmatrix(v, f)

    eigvals, eigfns = scipy.sparse.linalg.eigsh(
        L, numeigs + 1, M, sigma=0, which="LM", maxiter=100000
    )
    # Remove the zero eigenvalue.
    eigvals = eigvals[..., 1:]
    eigfns = eigfns[..., 1:]

    print(
        "largest eigval: ",
        eigvals.max().item(),
        ", smallest eigval: ",
        eigvals.min().item(),
    )
    return eigvals, eigfns


def sample_simplex_uniform(K, shape=(), dtype=torch.float32, device="cpu"):
    x = torch.sort(torch.rand(shape + (K,), dtype=dtype, device=device))[0]
    x = torch.cat(
        [
            torch.zeros(*shape, 1, dtype=dtype, device=device),
            x,
            torch.ones(*shape, 1, dtype=dtype, device=device),
        ],
        dim=-1,
    )
    diffs = x[..., 1:] - x[..., :-1]
    return diffs


def get_triangle_areas(v, f):
    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    return 0.5 * torch.norm(torch.linalg.cross(v1 - v0, v2 - v0), dim=1)


def create_eigfn(
    obj: str, idx: int, nsamples: int = 500000, upsample: int = 0, replace: bool = False
):
    data_dir = Path(__file__).parent / "data"
    if not replace and os.path.exists(data_dir / f"{obj}_eigfn{idx:03d}.npy"):
        print(f"{obj}_eigfn{idx:03d}.npy exists. Skipping.")
        return

    np.random.seed(777)

    v, f = igl.read_triangle_mesh(data_dir / f"{obj}.obj")
    if upsample > 0:
        v, f = igl.upsample(v, f, upsample)

    eigvals, eigfns = get_eigfn(v, f, numeigs=idx + 1)
    eigfns = torch.tensor(eigfns).float()
    v, f = torch.tensor(v).float(), torch.tensor(f).long()

    vals = eigfns[:, idx].clamp(min=0.0000)
    vals = torch.mean(vals[f], dim=1)
    vals = vals * get_triangle_areas(v, f)

    f_idx = torch.multinomial(vals, nsamples, replacement=True)
    barycoords = sample_simplex_uniform(2, (nsamples,), dtype=v.dtype, device=v.device)
    samples = torch.sum(v[f[f_idx]] * barycoords[..., None], axis=1)
    samples = samples.cpu().detach().numpy()

    with open(data_dir / f"{obj}_eigfn{idx:03d}.npy", "wb") as f:
        np.save(f, samples.astype(np.float32))
        print(f"Saved {data_dir / f'{obj}_eigfn{idx:03d}.npy'}.")


if __name__ == "__main__":
    create_eigfn("bunny", 9, upsample=3, replace=False)
    create_eigfn("bunny", 49, upsample=3, replace=False)
    create_eigfn("bunny", 99, upsample=3, replace=False)

    create_eigfn("spot", 9, upsample=3, replace=False)
    create_eigfn("spot", 49, upsample=3, replace=False)
    create_eigfn("spot", 99, upsample=3, replace=False)
