from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from .config import ExperimentConfig


@dataclass
class GraphBundle:
    adjacency: np.ndarray
    normalized_adjacency: np.ndarray
    laplacian_pe: np.ndarray
    chebyshev: np.ndarray


def _load_pickle_adjacency(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        try:
            payload = pickle.load(stream)
        except UnicodeDecodeError:
            stream.seek(0)
            payload = pickle.load(stream, encoding="latin1")
    if isinstance(payload, (tuple, list)) and len(payload) >= 3:
        payload = payload[2]
    if isinstance(payload, dict):
        for key in ("adj_mx", "adjacency", "A"):
            if key in payload:
                payload = payload[key]
                break
    return np.asarray(payload, dtype=np.float32)


def _load_edge_csv(path: Path, nodes: int, node_ids_path: Path | None) -> np.ndarray:
    adjacency = np.zeros((nodes, nodes), dtype=np.float32)
    id_map: dict[int, int] | None = None
    if node_ids_path is not None:
        identifiers = [int(value) for value in node_ids_path.read_text().strip().splitlines()]
        id_map = {identifier: index for index, identifier in enumerate(identifiers)}
    with path.open("r", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        first = next(reader, None)
        rows = [] if first is None else [first]
        rows.extend(reader)
    for row in rows:
        if len(row) < 2:
            continue
        try:
            source = int(float(row[0]))
            target = int(float(row[1]))
        except ValueError:
            continue
        if id_map is not None:
            if source not in id_map or target not in id_map:
                continue
            source, target = id_map[source], id_map[target]
        if 0 <= source < nodes and 0 <= target < nodes:
            adjacency[source, target] = 1.0
    return adjacency


def load_adjacency(config: ExperimentConfig, dataset_name: str) -> np.ndarray:
    spec = config.dataset(dataset_name)
    path = Path(spec["graph_path"])
    graph_format = spec["graph_format"]
    if graph_format == "dcrnn_pickle":
        adjacency = _load_pickle_adjacency(path)
    elif graph_format == "matrix_npy":
        adjacency = np.load(path).astype(np.float32)
    elif graph_format == "edge_csv":
        adjacency = _load_edge_csv(path, int(spec["nodes"]), spec.get("node_ids_path"))
    else:
        raise ValueError(f"Unsupported graph format: {graph_format}")
    expected = (int(spec["nodes"]), int(spec["nodes"]))
    if adjacency.shape != expected:
        raise ValueError(f"{dataset_name}: expected graph {expected}, found {adjacency.shape}")
    return adjacency


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    graph = adjacency.astype(np.float64) + np.eye(adjacency.shape[0], dtype=np.float64)
    degree = graph.sum(axis=1)
    degree = np.maximum(degree, 1.0e-5)
    inverse = 1.0 / np.sqrt(degree)
    return (inverse[:, None] * graph * inverse[None, :]).astype(np.float32)


def laplacian_positional_encoding(adjacency: np.ndarray, dimension: int = 32) -> np.ndarray:
    sym_graph = np.maximum(adjacency, adjacency.T).astype(np.float64)
    degree = sym_graph.sum(axis=1)
    inverse = np.zeros_like(degree)
    positive = degree > 0
    inverse[positive] = 1.0 / np.sqrt(degree[positive])
    normalized = sparse.eye(len(degree)) - sparse.diags(inverse) @ sparse.csr_matrix(sym_graph) @ sparse.diags(inverse)
    count = min(dimension + 1, adjacency.shape[0] - 1)
    try:
        _, vectors = eigsh(normalized, k=count, which="SM")
    except Exception:
        _, vectors = np.linalg.eigh(normalized.toarray())
        vectors = vectors[:, :count]
    vectors = vectors[:, 1 : dimension + 1]
    if vectors.shape[1] < dimension:
        vectors = np.pad(vectors, ((0, 0), (0, dimension - vectors.shape[1])))
    for column in range(vectors.shape[1]):
        anchor = np.argmax(np.abs(vectors[:, column]))
        if vectors[anchor, column] < 0:
            vectors[:, column] *= -1
    return vectors.astype(np.float32)


def chebyshev_basis(normalized_adjacency: np.ndarray, order: int = 3) -> np.ndarray:
    nodes = normalized_adjacency.shape[0]
    degree = normalized_adjacency.sum(axis=1)
    laplacian = np.diag(degree) - normalized_adjacency
    positive = degree > 0
    inverse = np.zeros_like(degree)
    inverse[positive] = 1.0 / np.sqrt(degree[positive])
    laplacian = inverse[:, None] * laplacian * inverse[None, :]
    eigenvalues = np.linalg.eigvals(laplacian)
    maximum = max(float(np.max(eigenvalues.real)), 1.0e-6)
    scaled = 2.0 * laplacian / maximum - np.eye(nodes)
    polynomials = [np.eye(nodes, dtype=np.float64)]
    if order > 1:
        polynomials.append(scaled)
    for _ in range(2, order):
        polynomials.append(2.0 * scaled @ polynomials[-1] - polynomials[-2])
    return np.stack(polynomials, axis=0).astype(np.float32)


def build_graph_bundle(config: ExperimentConfig, dataset_name: str) -> GraphBundle:
    cache_dir = config.path_for("data") / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{dataset_name}_graph.npz"
    if cache_path.exists():
        with np.load(cache_path) as archive:
            return GraphBundle(
                adjacency=archive["adjacency"],
                normalized_adjacency=archive["normalized_adjacency"],
                laplacian_pe=archive["laplacian_pe"],
                chebyshev=archive["chebyshev"],
            )
    adjacency = load_adjacency(config, dataset_name)
    normalized = normalize_adjacency(adjacency)
    bundle = GraphBundle(
        adjacency=adjacency,
        normalized_adjacency=normalized,
        laplacian_pe=laplacian_positional_encoding(adjacency),
        chebyshev=chebyshev_basis(normalized),
    )
    np.savez_compressed(
        cache_path,
        adjacency=bundle.adjacency,
        normalized_adjacency=bundle.normalized_adjacency,
        laplacian_pe=bundle.laplacian_pe,
        chebyshev=bundle.chebyshev,
    )
    return bundle
