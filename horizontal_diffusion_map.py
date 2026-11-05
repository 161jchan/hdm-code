"""Horizontal diffusion operator on precinct-district data.

Follows the horizontal part of Horizontal Diffusion Maps (Gao, 2019), with
alpha-normalization from Coifman and Lafon (2006).
"""

import logging
import random

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import kneighbors_graph

logger = logging.getLogger("horizontal_diffusion_map")


class HorizontalDiffusionMap:
    """Horizontal diffusion on the space of (precinct, district) pairs.

    A district is a data object and its precincts are the points on it. A
    horizontal step goes from (p, A) to (p, B): same precinct, different
    district. The shared precinct is the correspondence between A and B.

    Horizontal edges never leave a precinct, so this operator is reducible
    and has one eigenvalue equal to 1 per connected component. embed() is
    only informative once W is combined with a vertical operator, which is
    why W and D_W are exposed:

        T = omega * Dv^-1 Av + (1 - omega) * Dw^-1 W

    Attributes:
        W: symmetrized, alpha-normalized affinity matrix (sparse CSR)
        D_W: diagonal matrix of the row sums of W
        transition_matrix: row-stochastic Markov matrix P (sparse CSR)
        evals: leading eigenvalues of P, descending
        evecs: corresponding right eigenvectors of P
    """

    def __init__(
        self,
        data: list,
        alpha: float,
        bandwidth: float,
        name: str,
        num_evecs: int = 10,
        n_neighbors: int = 10,
    ):
        """Build the operator and compute its leading eigenpairs.

        Args:
            data: list of length num_precincts. data[j] has shape
                (num_districts_containing_j, num_precincts), and each row is
                the 0/1 encoding of one district that contains precinct j.
            alpha: density normalization exponent. 0 keeps the sampling
                density of the ensemble and 1 removes it.
            bandwidth: kernel rate b in W_ij = exp(-b * d(i, j)^2). Large b
                keeps only very similar districts. As b goes to 0 the walk
                becomes uniform over a point's neighbours.
            name: label for this run
            num_evecs: number of eigenpairs to compute
            n_neighbors: neighbours per point
        """
        self.name = f"hdm_{name}_{alpha}_{bandwidth}_{num_evecs}_{n_neighbors}"

        rows_list, cols_list, vals_list = [], [], []
        offset = 0

        for precinct_group in data:
            districts = np.asarray(precinct_group, dtype=bool)
            size = len(districts)

            # A precinct seen in only one district has no horizontal neighbours.
            if size <= 1:
                offset += size
                continue

            A = kneighbors_graph(
                districts,
                n_neighbors=min(n_neighbors, size - 1),
                metric="jaccard",
                include_self=False,
            )
            local_rows = np.repeat(np.arange(size), np.diff(A.indptr))
            local_cols = A.indices

            # Jaccard distance d(A, B) = |A symmetric difference B| / |A union B|
            X, Y = districts[local_rows], districts[local_cols]
            intersection = np.logical_and(X, Y).sum(axis=1)
            union = np.logical_or(X, Y).sum(axis=1)
            distances = 1 - intersection / union

            rows_list.append(offset + local_rows)
            cols_list.append(offset + local_cols)
            vals_list.append(np.exp(-bandwidth * distances**2))
            offset += size

        self.num_points = offset

        if rows_list:
            triples = (
                np.concatenate(vals_list),
                (np.concatenate(rows_list), np.concatenate(cols_list)),
            )
        else:
            triples = (np.array([], dtype=float), (np.array([], dtype=int),) * 2)

        W = csr_matrix(triples, shape=(offset, offset))
        W = (W + W.T) / 2

        # W_alpha = D^-alpha W D^-alpha
        d = np.asarray(W.sum(axis=1)).ravel()
        d[d == 0] = 1.0
        D_inv_alpha = diags(d ** (-alpha))
        W_alpha = (D_inv_alpha @ W @ D_inv_alpha).tocsr()

        # P = D_alpha^-1 W_alpha
        d_alpha = np.asarray(W_alpha.sum(axis=1)).ravel()
        d_alpha[d_alpha == 0] = 1.0
        P = (diags(1.0 / d_alpha) @ W_alpha).tocsr()

        # P is not symmetric, so diagonalize its symmetric conjugate. If
        # M v = lambda v then P w = lambda w, where w = D_alpha^(-1/2) v.
        inv_sqrt = d_alpha ** (-0.5)
        D_inv_sqrt = diags(inv_sqrt)
        M = D_inv_sqrt @ W_alpha @ D_inv_sqrt
        evals, evecs = eigsh(M, k=num_evecs, which="LA")
        evecs = evecs * inv_sqrt[:, None]

        order = evals.argsort()[::-1]
        self.evals = evals[order]
        self.evecs = evecs[:, order]

        self.W = W_alpha
        self.D_W = diags(d_alpha)
        self.transition_matrix = P

    def embed(self, n_dimensions: int = 2, t: float = 0.0) -> np.ndarray:
        """Return diffusion coordinates at time t, skipping the first eigenvector."""
        return self.evecs[:, 1 : n_dimensions + 1] * (
            self.evals[1 : n_dimensions + 1] ** t
        )

    def random_walk(self, idx: int = 0):
        """Yield the vertices of a random walk starting at idx."""
        P = self.transition_matrix
        while True:
            p = random.uniform(0, 1)
            start, end = P.indptr[idx], P.indptr[idx + 1]

            next_idx = None
            for i in range(start, end):
                p -= P.data[i]
                if p <= 0:
                    next_idx = P.indices[i]
                    break

            if next_idx is None and start < end:
                next_idx = P.indices[end - 1]

            if next_idx is None:
                return

            idx = next_idx
            yield idx
