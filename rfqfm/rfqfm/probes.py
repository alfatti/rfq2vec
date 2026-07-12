"""The recovery probe harness.

Each probe asks whether a trained model has recovered a specific piece of the
data-generating process, and reports a scalar that should climb with model
scale. The point of running these on the *simulator* rather than only on real
data is that we hold the ground truth: the realized word vectors
(oracle_token_vectors), the client tilts (oracle_clients), the attribute gain B
(oracle_attribute_gain), and the exact choice probabilities (event_truth).

Every probe takes the model artifact it needs plus an RfqDataset, so the harness
is testable without a GPU: feeding the oracle quantities back in as if they were
a perfect model must make each probe report near-perfect recovery. That
"perfect-model" path is the self-test at the bottom, and it is what runs in CI
here; the same functions point at real checkpoints on the box.

Implemented here:
  A  substitution geometry     -- model embedding Gram vs oracle Gram, top-k
  B  side-sense polysemy       -- buy/sell separation and side-axis coherence
  C  client-conditional tilt   -- recovery of u_k from the model's client tilt
  G  calibration               -- excess cross-entropy above the DGP floor

Probes that require a model's *predictive distribution* with context (the
regime/intensity test D, package-completion F, roll-down E) are metric functions
whose contract is documented where they'd slot in; they need decoded model
outputs and are wired once checkpoints exist.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contract import RfqDataset
from .floor import excess_over_floor


# ---------------------------------------------------------------------------
# oracle geometry
# ---------------------------------------------------------------------------

@dataclass
class OracleGeometry:
    token_id: np.ndarray      # (n_tok,)
    v: np.ndarray             # (n_tok, d) realized vectors at run start
    instr: np.ndarray         # (n_tok,) instrument_id per token
    sense: np.ndarray         # (n_tok,) side sense
    B: np.ndarray             # (p, d) attribute gain

    @property
    def direction(self) -> np.ndarray:
        n = np.linalg.norm(self.v, axis=1, keepdims=True)
        return self.v / np.clip(n, 1e-12, None)


def load_oracle_geometry(ds: RfqDataset) -> OracleGeometry:
    tv = ds.dim("oracle_token_vectors")
    tid = tv.column("token_id").to_numpy()
    v = np.asarray(tv.column("v").to_pylist(), dtype=np.float64)
    tm = ds.token_map()
    order_tid = tm.column("token_id").to_numpy()
    instr_of = tm.column("instrument_id").to_numpy()
    sense_of = tm.column("sense").to_numpy()
    lut_i = np.full(int(order_tid.max()) + 1, -1, np.int64)
    lut_s = np.full(int(order_tid.max()) + 1, -1, np.int64)
    lut_i[order_tid] = instr_of
    lut_s[order_tid] = sense_of
    B = np.asarray(ds.dim("oracle_attribute_gain").column("gain").to_pylist(),
                   dtype=np.float64)
    return OracleGeometry(token_id=tid, v=v, instr=lut_i[tid], sense=lut_s[tid], B=B)


def _subsample(n: int, max_n: int, seed: int = 0) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, max_n, replace=False)


# ---------------------------------------------------------------------------
# A -- substitution geometry
# ---------------------------------------------------------------------------

def _alignment_curve(S_true: np.ndarray, S_model: np.ndarray,
                     ks) -> dict:
    """Top-k subspace overlap between two similarity matrices over the same
    tokens. For each k, take the leading k eigenvectors of each and report the
    mean squared canonical correlation (1.0 = identical top-k subspace)."""
    wt, Ut = np.linalg.eigh(S_true)
    wm, Um = np.linalg.eigh(S_model)
    Ut, Um = Ut[:, ::-1], Um[:, ::-1]     # descending
    out = {}
    for k in ks:
        k = int(min(k, Ut.shape[1]))
        M = Ut[:, :k].T @ Um[:, :k]
        out[k] = float((M ** 2).sum() / k)
    return out


def probe_substitution_geometry(model_emb: np.ndarray, ds: RfqDataset,
                                ks=(1, 2, 4, 8, 16, 32, 64),
                                max_tokens: int = 4000) -> dict:
    """model_emb : (n_tok, h) the model's anchor-token embedding table, in the
    same token order as oracle_token_vectors.

    Compares the model's token-similarity geometry to the oracle's, top-k across
    a sweep of k. A small model recovers the leading directions then falls off;
    a large model tracks to high k."""
    geo = load_oracle_geometry(ds)
    idx = _subsample(len(geo.token_id), max_tokens)
    D = geo.direction[idx]
    E = model_emb[idx]
    E = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)
    S_true = D @ D.T
    S_model = E @ E.T
    off = ~np.eye(len(idx), dtype=bool)
    corr = float(np.corrcoef(S_true[off], S_model[off])[0, 1])
    return dict(n_tokens=len(idx),
                gram_offdiag_corr=corr,
                topk_alignment=_alignment_curve(S_true, S_model, ks))


# ---------------------------------------------------------------------------
# B -- side-sense polysemy
# ---------------------------------------------------------------------------

def _side_axis(emb: np.ndarray, instr: np.ndarray, sense: np.ndarray):
    """Per-instrument (buy - sell) deltas and their coherence. If side is a
    consistent sense direction, the deltas align (high mean pairwise cosine)."""
    buy = {int(i): r for r, (i, s) in enumerate(zip(instr, sense)) if s == 0}
    sell = {int(i): r for r, (i, s) in enumerate(zip(instr, sense)) if s == 1}
    common = sorted(set(buy) & set(sell))
    if len(common) < 2:
        return None
    deltas = np.array([emb[buy[i]] - emb[sell[i]] for i in common])
    dn = deltas / np.clip(np.linalg.norm(deltas, axis=1, keepdims=True), 1e-12, None)
    mean_dir = dn.mean(axis=0)
    coherence = float(np.linalg.norm(mean_dir))     # 1 = perfectly aligned deltas
    # separability: distance between buy and sell vs within-instrument scale
    en = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    sep = float(np.mean([1 - en[buy[i]] @ en[sell[i]] for i in common]))
    return dict(n_instruments=len(common), side_axis_coherence=coherence,
                buy_sell_separation=sep)


def probe_side_sense(model_emb: np.ndarray, ds: RfqDataset) -> dict:
    """Does the model keep buy and sell as distinct senses (rather than
    collapsing them to a shared instrument centroid)? Reports the model's
    side-axis coherence and buy/sell separation next to the oracle's."""
    geo = load_oracle_geometry(ds)
    model = _side_axis(model_emb, geo.instr, geo.sense)
    oracle = _side_axis(geo.v, geo.instr, geo.sense)
    return dict(model=model, oracle=oracle)


# ---------------------------------------------------------------------------
# C -- client-conditional substitution
# ---------------------------------------------------------------------------

def probe_client_tilt_recovery(model_tilt: np.ndarray, ds: RfqDataset) -> dict:
    """model_tilt : (K, d_model) the model's implied per-client tilt (e.g. its
    learned CLI embedding, or the mean logit shift it applies per client).

    Regresses the model tilt onto the true u_k via canonical correlation; the
    fraction of u_k variance the model's tilt can linearly reconstruct is the
    recovery score. A small model gets marginal client popularity (a scalar per
    client) but not the conditional rotation (the full d-vector u_k)."""
    oc = ds.dim("oracle_clients")
    u = np.asarray(oc.column("u").to_pylist(), dtype=np.float64)    # (K, d)
    T = np.asarray(model_tilt, dtype=np.float64)
    if T.shape[0] != u.shape[0]:
        raise ValueError(f"tilt has {T.shape[0]} clients, oracle has {u.shape[0]}")
    Uc = u - u.mean(0)
    Tc = T - T.mean(0)
    # canonical correlations between Tc and Uc
    Qt, _ = np.linalg.qr(Tc)
    Qu, _ = np.linalg.qr(Uc)
    s = np.linalg.svd(Qt.T @ Qu, compute_uv=False)
    ccs = np.clip(s, 0, 1)
    return dict(n_clients=int(u.shape[0]),
                mean_canonical_corr=float(ccs.mean()),
                top_canonical_corr=float(ccs.max()),
                recovered_dims=int((ccs > 0.5).sum()))


# ---------------------------------------------------------------------------
# G -- calibration against the DGP floor
# ---------------------------------------------------------------------------

def probe_calibration(model_logp: np.ndarray, event_ids: np.ndarray,
                      ds: RfqDataset) -> dict:
    """Excess cross-entropy above the given-latent floor, overall. See floor.py
    for the two-level decomposition; this is the headline Test-G number."""
    return excess_over_floor(model_logp, event_ids, ds)


# ---------------------------------------------------------------------------
# perfect-model self-test (the CI path here; no GPU needed)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# scaling table -- the headline artifact
# ---------------------------------------------------------------------------

def scaling_table(reports: dict) -> dict:
    """Collapse per-rung probe outputs into one metric-vs-size table.

    reports : {rung_name: {"params": int, "A_substitution": .., "B_side_sense": ..,
                           "C_client_tilt": .., "G_calibration": ..}}
    Each rung's dict is what run_all_probes / perfect_model_selftest returns,
    plus a "params" transformer parameter count. Rows come out sorted by size so
    the recovery-climbs-with-scale story reads straight down the column."""
    rows = []
    for name, r in reports.items():
        A, B = r.get("A_substitution"), r.get("B_side_sense")
        C, G = r.get("C_client_tilt"), r.get("G_calibration")
        D = r.get("D_decomposition")
        hi_k = max(A["topk_alignment"]) if A else None
        coh_ratio = None
        if B and B.get("model") and B.get("oracle"):
            om = B["oracle"]["side_axis_coherence"]
            coh_ratio = (B["model"]["side_axis_coherence"] / om) if om else None
        rows.append(dict(
            rung=r.get("size", name),
            variant=r.get("variant"),
            params=r.get("params"),
            gram_corr=(A["gram_offdiag_corr"] if A else None),
            topk_alignment_hi=(A["topk_alignment"][hi_k] if A else None),
            hi_k=hi_k,
            attribute_recovery=(D["attribute_recovery"] if D else None),
            residual_recovery=(D["residual_recovery"] if D else None),
            side_coherence_ratio=coh_ratio,
            client_canon_corr=(C["mean_canonical_corr"] if C else None),
            excess_over_floor_nats=(G["excess_over_floor_one_nats"] if G else None),
        ))
    # sort by variant then size so the L0/L1 pairs read across at each rung
    rows.sort(key=lambda d: (str(d["variant"]), d["params"] is None, d["params"] or 0))
    return dict(rows=rows, columns=list(rows[0].keys()) if rows else [])


def render_scaling_table(table: dict) -> str:
    """A compact fixed-width rendering for logs / the console."""
    head = (f"{'rung':<8}{'variant':<11}{'params':>12}{'gram':>7}"
            f"{'attr':>7}{'resid':>7}{'u_k':>7}{'excessKL':>10}")
    lines = [head, "-" * len(head)]
    for r in table["rows"]:
        def f(x, w, p=3):
            return (f"{x:>{w}.{p}f}" if isinstance(x, float) else f"{str(x):>{w}}")
        lines.append(
            f"{str(r['rung']):<8}{str(r.get('variant')):<11}{str(r['params']):>12}"
            f"{f(r['gram_corr'],7)}{f(r['attribute_recovery'],7)}"
            f"{f(r['residual_recovery'],7)}{f(r['client_canon_corr'],7)}"
            f"{f(r['excess_over_floor_nats'],10)}")
    return "\n".join(lines)


def perfect_model_selftest(ds: RfqDataset) -> dict:
    """Feed the oracle quantities in as if they were a perfect model. Every
    probe must report near-perfect recovery; this validates the harness end to
    end without a trained checkpoint."""
    geo = load_oracle_geometry(ds)

    A = probe_substitution_geometry(geo.v.copy(), ds)
    B = probe_side_sense(geo.v.copy(), ds)

    oc = ds.dim("oracle_clients")
    u = np.asarray(oc.column("u").to_pylist(), dtype=np.float64)
    C = probe_client_tilt_recovery(u.copy(), ds)

    t = ds.scan("event_truth", columns=["event_id", "log_p_chosen"])
    eid = t.column("event_id").to_numpy()
    tlp = t.column("log_p_chosen").to_numpy()
    ok = np.isfinite(tlp)
    G = probe_calibration(tlp[ok], eid[ok], ds)

    return {"A_substitution": A, "B_side_sense": B,
            "C_client_tilt": C, "G_calibration": G}
