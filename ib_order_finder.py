# -*- coding: utf-8 -*-
"""
IB Order Finder
===============
Standalone module for Information-Bottleneck-based feature ordering.

Splits the v07 monolith into three independent layers:

  Layer 1 – Statistics (no ML dependency)
      _discretize, _h, _h_joint, _mi, _h_cond
      compute_chain_metrics(order, df)          <- evaluate any ordering

  Layer 2 – IBOrderFinder (numpy / networkx only, no GReaT)
      IBOrderFinder.fit(df)
      IBOrderFinder.get_order()
      IBOrderFinder.get_mi_dataframe()
      IBOrderFinder.get_cond_entropy_dataframe()
      IBOrderFinder.summary()                   <- human-readable report
      IBOrderFinder.plot_mi_matrix()            <- optional matplotlib heatmap
      IBOrderFinder.save(path) / .load(path)

  Layer 3 – FixedOrderGReaT (imports be_great only when instantiated)
      FixedOrderGReaT(fixed_order, **great_kwargs)
      FixedOrderGReaT.fit(df)
      FixedOrderGReaT.sample(...)

Convenience entry point:
  find_order(df, n_bins=100, mi_threshold_quantile=0.25) -> List[str]

Typical usage::

    from ib_order_finder import IBOrderFinder, FixedOrderGReaT, compute_chain_metrics

    # 1. Find ordering
    finder = IBOrderFinder(n_bins=100, mi_threshold_quantile=0.25)
    finder.fit(train_df)
    order = finder.get_order()          # ['lon', 'lat', 'state_code', ...]
    print(finder.summary())

    # 2. Evaluate any ordering
    metrics = compute_chain_metrics(order, train_df)
    print(f"chain_CE={metrics['chain_ce']:.3f}  chain_MI={metrics['chain_mi']:.3f}")

    # 3. Train GReaT with the derived order
    model = FixedOrderGReaT(fixed_order=order, llm="gpt2-medium", epochs=100)
    model.fit(train_df)
    syn_df = model.sample(n_samples=len(train_df))
"""

import heapq
import json
import logging
import typing as tp

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Layer 1: Low-level entropy / MI statistics
# ──────────────────────────────────────────────────────────────────────────────

def _discretize(series: pd.Series, n_bins: int) -> np.ndarray:
    """Discretise a column into integer codes.

    - Float or high-cardinality int  ->  equal-frequency bins (qcut).
    - Categorical / low-cardinality  ->  direct factorize.

    NaN / out-of-bin values are mapped to sentinel -1 and excluded from all
    entropy computations.
    """
    is_float = pd.api.types.is_float_dtype(series)
    is_high_card_int = (
        pd.api.types.is_integer_dtype(series) and series.nunique() > 2 * n_bins
    )
    if is_float or is_high_card_int:
        binned = pd.qcut(series, q=n_bins, labels=False, duplicates="drop")
        return binned.fillna(-1).astype(int).values
    codes, _ = pd.factorize(series)
    return codes.astype(int)


def _h(x: np.ndarray) -> float:
    """Shannon entropy H(X) in bits (ignores sentinel -1)."""
    x = x[x >= 0]
    if len(x) == 0:
        return 0.0
    _, c = np.unique(x, return_counts=True)
    p = c / c.sum()
    return float(-np.dot(p, np.log2(p + 1e-15)))


def _h_joint(x: np.ndarray, y: np.ndarray) -> float:
    """Joint entropy H(X, Y) in bits."""
    mask = (x >= 0) & (y >= 0)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return 0.0
    pairs = np.stack([x, y], axis=1)
    _, c = np.unique(pairs, axis=0, return_counts=True)
    p = c / c.sum()
    return float(-np.dot(p, np.log2(p + 1e-15)))


def _mi(x: np.ndarray, y: np.ndarray) -> float:
    """Mutual information I(X; Y) = H(X) + H(Y) - H(X,Y) in bits (clamped >= 0)."""
    return max(0.0, _h(x) + _h(y) - _h_joint(x, y))


def _h_cond(x: np.ndarray, y: np.ndarray) -> float:
    """Conditional entropy H(X | Y) = H(X,Y) - H(Y) in bits (clamped >= 0)."""
    return max(0.0, _h_joint(x, y) - _h(y))


def compute_chain_metrics(
    order: tp.List[str],
    df: pd.DataFrame,
    n_bins: int = 100,
) -> tp.Dict[str, float]:
    """Compute chain-CE and chain-MI for a given feature ordering.

    Definitions (same as orderings.json):
      chain_CE  = sum_{i>0} H(X_{order[i]} | X_{order[0]}, ..., X_{order[i-1]})
                  approximated as sum of pairwise H(X_i | X_{i-1})
      chain_MI  = sum_{i>0} I(X_{order[i]} ; X_{order[i-1]})

    Lower chain_CE  -> each feature is more predictable given its predecessor.
    Higher chain_MI -> successive features share more information.

    Args:
        order:  Feature names in generation order.
        df:     DataFrame containing those columns.
        n_bins: Bins for continuous feature discretisation.

    Returns:
        Dict with keys 'chain_ce' and 'chain_mi'.
    """
    cols = [c for c in order if c in df.columns]
    if len(cols) < 2:
        return {"chain_ce": 0.0, "chain_mi": 0.0}

    disc = {c: _discretize(df[c], n_bins) for c in cols}
    chain_ce = 0.0
    chain_mi = 0.0
    for i in range(1, len(cols)):
        prev, cur = cols[i - 1], cols[i]
        chain_ce += _h_cond(disc[cur], disc[prev])
        chain_mi += _mi(disc[prev], disc[cur])
    return {"chain_ce": round(chain_ce, 6), "chain_mi": round(chain_mi, 6)}


# ──────────────────────────────────────────────────────────────────────────────
# Layer 2: IBOrderFinder
# ──────────────────────────────────────────────────────────────────────────────

class IBOrderFinder:
    """Derive the optimal feature generation order directly from training data.

    The ordering answers: *"In what sequence should a language model generate
    features so that each feature is maximally constrained (low conditional
    entropy) given the features already generated?"*

    Algorithm
    ---------
    1. Discretise continuous features via equal-frequency binning.
    2. Compute pairwise MI:  I(X_i ; X_j) = H(X_i) + H(X_j) - H(X_i, X_j)
       and conditional entropies:  CE[i,j] = H(X_j | X_i).
    3. Threshold edges by MI significance (quantile on off-diagonal values).
    4. Add directed edge i -> j  iff  CE[i,j] < CE[j,i]
       (X_i determines X_j more than vice versa).
    5. Topological sort with max-entropy tiebreaking:
       - DAG:    Kahn's algorithm.
       - Cyclic: SCC condensation -> topological sort of condensed DAG.
    6. Append disconnected nodes by entropy descending.

    Attributes (populated after fit())
    -----------------------------------
    feature_names_  : List[str]
    entropy_        : np.ndarray  shape (m,)         H[i] in bits
    mi_matrix_      : np.ndarray  shape (m, m)       MI[i,j] in bits
    cond_entropy_   : np.ndarray  shape (m, m)       CE[i,j] = H(X_j|X_i)
    optimal_order_  : List[str]
    """

    def __init__(
        self,
        n_bins: int = 100,
        mi_threshold_quantile: float = 0.25,
    ) -> None:
        """
        Args:
            n_bins:
                Equal-frequency bins for continuous feature discretisation.
                Default 100 ensures lat/lon (continuous) entropy is not capped
                below state_code (~5.67 bits).  Requires n_rows >> n_bins.
            mi_threshold_quantile:
                Quantile of off-diagonal MI values used as edge-activation
                threshold.  0.25 retains the top 75% of pairs.  Increase to
                0.5 for a sparser graph, decrease toward 0 for a denser one.
        """
        self.n_bins = n_bins
        self.mi_threshold_quantile = mi_threshold_quantile

        self.feature_names_: tp.Optional[tp.List[str]] = None
        self.entropy_:       tp.Optional[np.ndarray]   = None
        self.mi_matrix_:     tp.Optional[np.ndarray]   = None
        self.cond_entropy_:  tp.Optional[np.ndarray]   = None
        self.optimal_order_: tp.Optional[tp.List[str]] = None
        self._mi_thresh_:    tp.Optional[float]        = None
        self._graph_edges_:  tp.Optional[tp.List[tp.Tuple]] = None  # (i, j, mi)

    # ── public API ─────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "IBOrderFinder":
        """Compute pairwise MI and derive the optimal generation order.

        Args:
            df: Training DataFrame; all columns are treated as features.

        Returns:
            self  (fluent interface: finder.fit(df).get_order())
        """
        features = list(df.columns)
        m = len(features)
        if m < 2:
            raise ValueError(
                f"IBOrderFinder requires at least 2 features, got {m}."
            )
        self.feature_names_ = features

        # Step 1: Discretise all features
        disc = {f: _discretize(df[f], self.n_bins) for f in features}

        # Step 2: Marginal entropies, pairwise MI, conditional entropies
        H  = np.array([_h(disc[f]) for f in features])
        MI = np.zeros((m, m))
        CE = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                MI[i, j] = _mi(disc[features[i]], disc[features[j]])
                CE[i, j] = _h_cond(disc[features[j]], disc[features[i]])

        self.entropy_     = H
        self.mi_matrix_   = MI
        self.cond_entropy_ = CE

        logger.info("IBOrderFinder: marginal entropies (bits)")
        for i, f in enumerate(features):
            logger.info(f"  H({f}) = {H[i]:.4f}")

        # Step 3: Directed MI graph
        off_diag  = MI[~np.eye(m, dtype=bool)]
        mi_thresh = float(np.quantile(off_diag, self.mi_threshold_quantile))
        self._mi_thresh_ = mi_thresh
        logger.info(
            f"MI edge threshold (q={self.mi_threshold_quantile}): {mi_thresh:.4f} bits"
        )

        G = nx.DiGraph()
        G.add_nodes_from(range(m))
        active_edges = []
        for i in range(m):
            for j in range(m):
                if i == j or MI[i, j] <= mi_thresh:
                    continue
                if CE[i, j] < CE[j, i]:
                    G.add_edge(i, j, weight=float(MI[i, j]))
                    active_edges.append((i, j, float(MI[i, j])))

        self._graph_edges_ = active_edges
        logger.info("Directed edges (i -> j means: generate i before j):")
        for u, v, w in sorted(active_edges, key=lambda e: -e[2]):
            logger.info(
                f"  {features[u]:>14} -> {features[v]:<14}  "
                f"MI={w:.4f}  "
                f"H({features[v]}|{features[u]})={CE[u,v]:.4f} < "
                f"H({features[u]}|{features[v]})={CE[v,u]:.4f}"
            )

        # Step 4: Topological sort (handles cycles via SCC condensation)
        order_idx = self._topo_sort(G, H)

        # Step 5: Append disconnected nodes by entropy descending
        placed = set(order_idx)
        tail   = sorted([i for i in range(m) if i not in placed], key=lambda x: -H[x])
        order_idx.extend(tail)

        self.optimal_order_ = [features[i] for i in order_idx]
        logger.info(f"Optimal order: {self.optimal_order_}")
        return self

    def get_order(self) -> tp.List[str]:
        """Return the optimal feature order as a list of column names."""
        self._check_fitted()
        return list(self.optimal_order_)

    def get_mi_dataframe(self) -> pd.DataFrame:
        """Return pairwise MI matrix (bits) as a labelled DataFrame."""
        self._check_fitted()
        return pd.DataFrame(
            self.mi_matrix_, index=self.feature_names_, columns=self.feature_names_
        )

    def get_cond_entropy_dataframe(self) -> pd.DataFrame:
        """Return conditional entropy matrix CE[i,j] = H(X_j | X_i) as DataFrame."""
        self._check_fitted()
        return pd.DataFrame(
            self.cond_entropy_, index=self.feature_names_, columns=self.feature_names_
        )

    def summary(self) -> str:
        """Return a human-readable summary of the ordering result."""
        self._check_fitted()
        lines = [
            "=" * 56,
            "IBOrderFinder Summary",
            "=" * 56,
            f"  n_bins               = {self.n_bins}",
            f"  mi_threshold_quantile= {self.mi_threshold_quantile}",
            f"  MI edge threshold    = {self._mi_thresh_:.4f} bits",
            "",
            "Marginal entropies (bits):",
        ]
        for i, f in enumerate(self.feature_names_):
            lines.append(f"  H({f}) = {self.entropy_[i]:.4f}")
        lines += [
            "",
            "Active directed edges (i -> j: generate i before j):",
        ]
        if self._graph_edges_:
            for u, v, w in sorted(self._graph_edges_, key=lambda e: -e[2]):
                lines.append(
                    f"  {self.feature_names_[u]} -> {self.feature_names_[v]}"
                    f"  (MI={w:.4f})"
                )
        else:
            lines.append("  (no active edges — graph is empty)")
        lines += [
            "",
            "Optimal generation order:",
            "  " + " -> ".join(self.optimal_order_),
            "=" * 56,
        ]
        return "\n".join(lines)

    def plot_mi_matrix(
        self,
        ax=None,
        title: str = "Pairwise Mutual Information (bits)",
        cmap: str = "YlOrRd",
        annotate: bool = True,
    ):
        """Plot the MI matrix as a heatmap.  Requires matplotlib.

        Args:
            ax:       Existing matplotlib Axes.  Creates a new figure if None.
            title:    Plot title.
            cmap:     Colormap name.
            annotate: Whether to print values in each cell.

        Returns:
            The matplotlib Axes object.
        """
        self._check_fitted()
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plot_mi_matrix().")

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))

        im = ax.imshow(self.mi_matrix_, cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, label="MI (bits)")
        ax.set_xticks(range(len(self.feature_names_)))
        ax.set_yticks(range(len(self.feature_names_)))
        ax.set_xticklabels(self.feature_names_, rotation=45, ha="right")
        ax.set_yticklabels(self.feature_names_)
        ax.set_title(title)

        if annotate:
            for i in range(len(self.feature_names_)):
                for j in range(len(self.feature_names_)):
                    ax.text(
                        j, i, f"{self.mi_matrix_[i, j]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="black" if self.mi_matrix_[i, j] < self.mi_matrix_.max() * 0.6
                        else "white",
                    )
        return ax

    def save(self, path: str) -> None:
        """Save fitted state to a JSON file.

        Args:
            path: File path (should end in .json).
        """
        self._check_fitted()
        data = {
            "n_bins":                self.n_bins,
            "mi_threshold_quantile": self.mi_threshold_quantile,
            "feature_names_":        self.feature_names_,
            "entropy_":              self.entropy_.tolist(),
            "mi_matrix_":            self.mi_matrix_.tolist(),
            "cond_entropy_":         self.cond_entropy_.tolist(),
            "optimal_order_":        self.optimal_order_,
            "_mi_thresh_":           self._mi_thresh_,
            "_graph_edges_":         self._graph_edges_,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"IBOrderFinder state saved to {path}")

    @classmethod
    def load(cls, path: str) -> "IBOrderFinder":
        """Restore a fitted IBOrderFinder from a JSON file.

        Args:
            path: Path written by :meth:`save`.

        Returns:
            Fitted IBOrderFinder instance (ready to call get_order() etc.).
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        obj = cls(
            n_bins=data["n_bins"],
            mi_threshold_quantile=data["mi_threshold_quantile"],
        )
        obj.feature_names_  = data["feature_names_"]
        obj.entropy_        = np.array(data["entropy_"])
        obj.mi_matrix_      = np.array(data["mi_matrix_"])
        obj.cond_entropy_   = np.array(data["cond_entropy_"])
        obj.optimal_order_  = data["optimal_order_"]
        obj._mi_thresh_     = data["_mi_thresh_"]
        obj._graph_edges_   = [tuple(e) for e in data["_graph_edges_"]]
        logger.info(f"IBOrderFinder state loaded from {path}")
        return obj

    # ── private helpers ────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self.optimal_order_ is None:
            raise RuntimeError("Call fit() before using this method.")

    def _topo_sort(self, G: nx.DiGraph, H: np.ndarray) -> tp.List[int]:
        """Topological sort with max-entropy tiebreaking and SCC condensation."""
        if nx.is_directed_acyclic_graph(G):
            # Regular DAG: node IDs are 0..m-1, index directly into H.
            h_map = {n: float(H[n]) for n in G.nodes()}
            return self._kahn_mapped(G, h_map)

        logger.info("Graph has cycles; condensing SCCs.")
        sccs = list(nx.strongly_connected_components(G))
        G_dag = nx.condensation(G, scc=sccs)
        # Build entropy map for condensed nodes.
        # Sort node IDs explicitly — do not rely on dict iteration order.
        scc_H_map = {
            k: max(H[n] for n in G_dag.nodes[k]["members"])
            for k in G_dag.nodes
        }
        scc_order = self._kahn_mapped(G_dag, scc_H_map)
        order_idx: tp.List[int] = []
        for scc_id in scc_order:
            members = sorted(G_dag.nodes[scc_id]["members"], key=lambda x: -H[x])
            order_idx.extend(members)
        return order_idx

    @staticmethod
    def _kahn_mapped(
        G: nx.DiGraph,
        h_map: tp.Dict[int, float],
    ) -> tp.List[int]:
        """Kahn's topological sort with max-entropy priority.

        Args:
            G:     Directed acyclic graph whose nodes are keys of h_map.
            h_map: Entropy (or representative entropy) for each node.
        """
        in_deg = dict(G.in_degree())
        # Max-heap: negate so highest-entropy node pops first.
        # Second element breaks ties deterministically (node ID).
        heap: tp.List[tp.Tuple[float, int]] = [
            (-h_map.get(n, 0.0), n)
            for n in G.nodes()
            if in_deg[n] == 0
        ]
        heapq.heapify(heap)
        order: tp.List[int] = []
        while heap:
            _, node = heapq.heappop(heap)
            order.append(node)
            for succ in G.successors(node):
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    heapq.heappush(heap, (-h_map.get(succ, 0.0), succ))
        return order


# ──────────────────────────────────────────────────────────────────────────────
# Layer 3: FixedOrderGReaT
# ──────────────────────────────────────────────────────────────────────────────

class FixedOrderGReaT:
    """GReaT wrapper that trains with a fixed column serialisation order.

    During the default GReaT ``fit()``, ``GReaTDataset._getitem`` randomly
    permutes columns for every training sample.  This class monkey-patches
    ``be_great.great.GReaTDataset`` with a fixed-order variant for the duration
    of ``fit()``, then restores the original class in a ``finally`` block.

    ``conditional_col`` is set to ``fixed_order[0]`` so that ``_legacy_sample``
    begins generation from the first feature, matching the training distribution.

    Args:
        fixed_order: Column names in the desired generation order.
        **kwargs:    Forwarded verbatim to ``GReaT.__init__``.

    Example::

        model = FixedOrderGReaT(
            fixed_order=["lon", "lat", "state_code", "bird", "lat_zone"],
            llm="gpt2-medium",
            epochs=100,
            batch_size=32,
            efficient_finetuning="lora",
        )
        model.fit(train_df)
        syn_df = model.sample(n_samples=16320, temperature=0.8, k=100)
    """

    def __init__(self, fixed_order: tp.List[str], **kwargs) -> None:
        # Lazy import: keeps this module usable without be_great installed
        try:
            import be_great.great as _great_module
            from be_great import GReaT
        except ImportError as exc:
            raise ImportError(
                "be_great is required for FixedOrderGReaT. "
                "Install it or use IBOrderFinder standalone."
            ) from exc

        self._great_module = _great_module
        self._model = GReaT(**kwargs)
        self.fixed_order: tp.List[str] = list(fixed_order)

    def fit(
        self,
        data: tp.Union["pd.DataFrame", "np.ndarray"],
        column_names: tp.Optional[tp.List[str]] = None,
        conditional_col: tp.Optional[str] = None,
        resume_from_checkpoint: tp.Union[bool, str] = False,
    ):
        """Fine-tune with the fixed column order throughout all epochs.

        Args:
            data:                   Training data.
            column_names:           Column names when data is an ndarray.
            conditional_col:        Start feature for legacy sampling.
                                    Defaults to ``fixed_order[0]``.
            resume_from_checkpoint: Passed through to GReaT.
        """
        from be_great.great_utils import _array_to_dataframe

        if not isinstance(data, pd.DataFrame):
            data = _array_to_dataframe(data, columns=column_names)
            column_names = None

        # Reorder DataFrame columns to match fixed_order (puts known cols first)
        ordered = [c for c in self.fixed_order if c in data.columns]
        rest    = [c for c in data.columns   if c not in ordered]
        data    = data[ordered + rest]

        _fixed    = self.fixed_order
        _orig_cls = self._great_module.GReaTDataset

        class _FixedOrderDataset(_orig_cls):
            def _getitem(self, key, decoded=True, **kwargs):
                row       = self._data.fast_slice(key, 1)
                col_names = row.column_names
                idx_map   = {name: i for i, name in enumerate(col_names)}
                idx       = [idx_map[c] for c in _fixed if c in idx_map]
                text      = ", ".join(
                    f"{col_names[i]} is "
                    f"{self._format_value(row.columns[i].to_pylist()[0])}"
                    for i in idx
                )
                return self.tokenizer(text, padding=True)

        self._great_module.GReaTDataset = _FixedOrderDataset
        try:
            return self._model.fit(
                data,
                column_names=None,
                conditional_col=conditional_col or self.fixed_order[0],
                resume_from_checkpoint=resume_from_checkpoint,
                random_conditional_col=False,
            )
        finally:
            self._great_module.GReaTDataset = _orig_cls

    def sample(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """Generate synthetic samples using legacy_sample.

        Args:
            n_samples: Number of rows to generate.
            **kwargs:  Forwarded to GReaT.sample()
                       (temperature, k, max_length, guided_sampling, ...).
        """
        return self._model.sample(n_samples=n_samples, **kwargs)

    def save(self, path: str) -> None:
        """Save the underlying GReaT model."""
        self._model.save(path)

    @property
    def train_hyperparameters(self) -> dict:
        return self._model.train_hyperparameters


# ──────────────────────────────────────────────────────────────────────────────
# Convenience entry point
# ──────────────────────────────────────────────────────────────────────────────

def find_order(
    df: pd.DataFrame,
    n_bins: int = 100,
    mi_threshold_quantile: float = 0.25,
    verbose: bool = True,
) -> tp.List[str]:
    """One-liner: fit IBOrderFinder and return the optimal feature order.

    Args:
        df:                     Training DataFrame.
        n_bins:                 Discretisation bins for continuous features.
        mi_threshold_quantile:  Edge activation quantile threshold.
        verbose:                Print the summary report.

    Returns:
        List of column names in optimal generation order.

    Example::

        order = find_order(train_df)
        # ['lon', 'lat', 'state_code', 'bird', 'lat_zone']
    """
    finder = IBOrderFinder(n_bins=n_bins, mi_threshold_quantile=mi_threshold_quantile)
    finder.fit(df)
    if verbose:
        print(finder.summary())
    return finder.get_order()
