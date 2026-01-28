from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import networkx as nx
from scipy import sparse
from scipy.sparse import linalg as spla

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from .graphs import compute_condensation


@dataclass
class Spectrum:
    evals: np.ndarray          # complex eigenvalues of directed adjacency (possibly truncated)
    svals: np.ndarray          # singular values of adjacency
    evals_sym: np.ndarray      # eigenvalues of symmetrized adjacency


def _adjacency_sparse(dag: nx.DiGraph) -> sparse.csr_matrix:
    nodes = list(dag.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    rows, cols = [], []
    for u, v in dag.edges():
        rows.append(idx[u])
        cols.append(idx[v])
    data = np.ones(len(rows), dtype=float)
    n = len(nodes)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def compute_spectrum(
    G: nx.MultiDiGraph,
    n_eigs: int = 64,
) -> Spectrum:
    cond = compute_condensation(G)
    dag = cond.dag
    A = _adjacency_sparse(dag)
    n = A.shape[0]
    # eigsh/eigs require k < n
    k = int(min(n_eigs, max(1, n - 2)))

    # Directed eigenvalues
    if n <= 256:
        evals = np.linalg.eigvals(A.toarray())
        evals = evals[np.argsort(-np.abs(evals))][:n_eigs]
    else:
        try:
            evals = spla.eigs(A, k=k, which="LM", return_eigenvectors=False, tol=1e-3, maxiter=5000)
        except Exception:
            evals = np.array([], dtype=complex)

    # Singular values
    if n <= 256:
        svals = np.linalg.svd(A.toarray(), compute_uv=False)
        svals = np.sort(svals)[::-1][:n_eigs]
    else:
        try:
            svals = spla.svds(A, k=k, return_singular_vectors=False, tol=1e-3)
            svals = np.sort(svals)[::-1]
        except Exception:
            svals = np.array([], dtype=float)

    # Symmetrized eigenvalues
    As = (A + A.T) * 0.5
    if n <= 256:
        evals_sym = np.linalg.eigvals(As.toarray())
        evals_sym = evals_sym[np.argsort(-np.abs(evals_sym))][:n_eigs]
    else:
        try:
            evals_sym = spla.eigs(As, k=k, which="LM", return_eigenvectors=False, tol=1e-3, maxiter=5000)
        except Exception:
            evals_sym = np.array([], dtype=complex)

    return Spectrum(evals=np.array(evals), svals=np.array(svals), evals_sym=np.array(evals_sym))


@dataclass
class ClusterResult:
    k: int
    labels: np.ndarray
    centers: np.ndarray
    score: float
    method: str


def _clean_1d(values: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    values = np.array(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    values = values[np.abs(values) > eps]
    # Sort descending (typical for singular values)
    values = np.sort(values)[::-1]
    return values


def cluster_values(
    values: np.ndarray,
    k: int,
    method: str = "gmm",
    random_state: int = 0,
) -> ClusterResult:
    x = values.reshape(-1, 1)
    if len(values) < k + 1:
        return ClusterResult(k=k, labels=np.zeros(len(values), dtype=int), centers=np.array([]), score=float("-inf"), method=method)

    if method == "kmeans":
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(x)
        centers = km.cluster_centers_.reshape(-1)
        score = silhouette_score(x, labels) if k >= 2 else float("-inf")
        return ClusterResult(k=k, labels=labels, centers=centers, score=float(score), method=method)

    if method == "gmm":
        # reg_covar mitigates singular covariance when values duplicate
        gmm = GaussianMixture(n_components=k, random_state=random_state, n_init=5, max_iter=500, reg_covar=1e-6)
        try:
            gmm.fit(x)
            labels = gmm.predict(x)
            centers = gmm.means_.reshape(-1)
            score = -float(gmm.bic(x))  # larger is better
        except Exception:
            labels = np.zeros(len(values), dtype=int)
            centers = np.array([])
            score = float("-inf")
        return ClusterResult(k=k, labels=labels, centers=centers, score=score, method=method)

    raise ValueError(f"Unknown method: {method}")


def choose_k_bands(
    values: np.ndarray,
    k_candidates: Sequence[int] = (2, 3, 4, 5),
    method: str = "gmm",
    random_state: int = 0,
) -> ClusterResult:
    values = _clean_1d(values)
    if len(values) == 0:
        return ClusterResult(k=0, labels=np.array([], dtype=int), centers=np.array([]), score=float("-inf"), method=method)

    # Limit k by number of distinct values (avoid degenerate clustering)
    distinct = len(np.unique(np.round(values, 12)))
    k_candidates = [int(k) for k in k_candidates if 2 <= int(k) <= max(2, min(distinct, len(values) - 1))]
    if not k_candidates:
        # fall back to k=2
        k_candidates = [2] if len(values) >= 3 else [1]

    best: Optional[ClusterResult] = None
    for k in k_candidates:
        res = cluster_values(values, k=k, method=method, random_state=random_state)
        if best is None or (res.score > best.score):
            best = res
    assert best is not None
    return best


@dataclass
class GenerationTestResult:
    spectrum: Spectrum
    clustering_svals: Optional[ClusterResult]
    clustering_evals_sym: Optional[ClusterResult]


def generation_band_test(
    G: nx.MultiDiGraph,
    n_vals: int = 64,
    k_candidates: Sequence[int] = (2, 3, 4, 5),
    method: str = "gmm",
    random_state: int = 0,
) -> GenerationTestResult:
    spec = compute_spectrum(G, n_eigs=int(n_vals))

    clustering_svals = None
    if spec.svals.size > 0:
        vals = spec.svals[: int(n_vals)]
        clustering_svals = choose_k_bands(vals, k_candidates=k_candidates, method=method, random_state=random_state)

    clustering_sym = None
    if spec.evals_sym.size > 0:
        vals = np.abs(spec.evals_sym[: int(n_vals)])
        clustering_sym = choose_k_bands(vals, k_candidates=k_candidates, method=method, random_state=random_state)

    return GenerationTestResult(spectrum=spec, clustering_svals=clustering_svals, clustering_evals_sym=clustering_sym)
