"""Bridge from a trained checkpoint to the numpy arrays the probes consume.

The probes themselves are torch-free and take arrays; this module is the only
place that loads a model and runs a forward pass, so it stays on the GPU box.
Three extractions, one per probe family:

  * anchor_embedding_table -> (n_tok, hidden), in oracle-token order, for the
    geometry / side-sense probes (A, B). For the factorized embedding this is
    the rebuilt anchor table; for a plain table it is the TOK slice.

  * client_tilt_table -> (K, hidden), in oracle-client order, for the
    client-conditional probe (C). This is the CLI embedding -- the rows where
    the model is free to store each client's tilt u_k.

  * anchor_logprobs -> (logp, event_id) over a held-out corpus, for calibration
    (G). Note the model scores over the *full* vocabulary while the generator's
    log_p_chosen is over the mandate-eligible candidate set, so the model also
    has to learn the mask from observables -- that cost shows up honestly in the
    excess over the floor.
"""
from __future__ import annotations

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("rfqfm.extract needs torch; run on the GPU box") from e

import numpy as np

from .tokenize import FAMILY
from .vocab import FmVocab

_TOK = FAMILY["TOK"]


@torch.no_grad()
def anchor_embedding_table(model, vocab: FmVocab) -> np.ndarray:
    """(n_tok, hidden) anchor embeddings in ascending-token order (matches
    oracle_token_vectors)."""
    emb = model.get_input_embeddings()
    fam = vocab.families["TOK"]
    if hasattr(emb, "anchor_table"):          # FactorizedTokenEmbedding
        tab = emb.anchor_table().detach().float().cpu().numpy()
    else:                                     # plain nn.Embedding: slice TOK rows
        ids = torch.arange(fam.offset, fam.offset + fam.size,
                           device=emb.weight.device)
        tab = emb(ids).detach().float().cpu().numpy()
    return tab


@torch.no_grad()
def client_tilt_table(model, vocab: FmVocab) -> np.ndarray:
    """(K, hidden) CLI-family embeddings in ascending-client order (matches
    oracle_clients)."""
    emb = model.get_input_embeddings()
    fam = vocab.families["CLI"]
    device = getattr(emb, "weight", next(model.parameters())).device \
        if hasattr(emb, "weight") else next(model.parameters()).device
    ids = torch.arange(fam.offset, fam.offset + fam.size, device=device)
    # both plain and factorized embeddings return CLI rows from the plain table
    if hasattr(emb, "plain"):
        rows = emb.plain(ids)
    else:
        rows = emb(ids)
    return rows.detach().float().cpu().numpy()


@torch.no_grad()
def anchor_logprobs(model, loader, device) -> tuple:
    """Run the model over a held-out corpus and collect, at every position whose
    next token is an anchor, the model's log-prob of that realized next token
    and its event_id. Returns (logp: np.ndarray, event_id: np.ndarray)."""
    model.eval()
    logps, evids = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        fam = batch["family"].to(device)
        ev = batch["event_id"].to(device)
        logits = model(input_ids=ids).logits         # (B, T, V)
        logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = ids[:, 1:]
        tgt_fam = fam[:, 1:]
        tgt_ev = ev[:, 1:]
        chosen = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # (B, T-1)
        mask = tgt_fam == _TOK
        logps.append(chosen[mask].detach().float().cpu().numpy())
        evids.append(tgt_ev[mask].detach().cpu().numpy().astype(np.uint64))
    return np.concatenate(logps), np.concatenate(evids)
