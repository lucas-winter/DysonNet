import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dysonnet.DysonNQS as mnqs
from dysonnet.custom_operator import TurboTurboOperator, _embed_windows_into_full, get_idx
from dysonnet.utils import setup_system_long_range_ising


jax.config.update("jax_enable_x64", False)

FAST_MODE = os.environ.get("PYTEST_FAST") == "1"

LOCAL_UPDATE_CASES = [
    dict(
        L=12,
        width=2,
        token_size=1,
        embedding_dim=8,
        hidden_dim=8,
        n_blocks=1,
        s4_states=4,
        seed=0,
    ),
    dict(
        L=12,
        width=2,
        token_size=2,
        embedding_dim=12,
        hidden_dim=12,
        n_blocks=1,
        s4_states=4,
        seed=1,
    ),
    dict(
        L=16,
        width=2,
        token_size=4,
        embedding_dim=16,
        hidden_dim=16,
        n_blocks=1,
        s4_states=8,
        seed=2,
    ),
]

FAST_LOCAL_UPDATE_CASES = [LOCAL_UPDATE_CASES[0]]
LOCAL_UPDATE_CASES_TO_RUN = (
    FAST_LOCAL_UPDATE_CASES if FAST_MODE else LOCAL_UPDATE_CASES
)


def _build_test_model(
    L: int,
    token_size: int,
    width: int,
    *,
    embedding_dim: int,
    hidden_dim: int,
    n_blocks: int,
    s4_states: int,
):
    s4_seq_len = L // token_size
    return mnqs.DysonNetFactory.create_model(
        token_size=token_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
        use_complex=True,
        use_logcosh_output=True,
        logcosh_hidden=2,
        conv_kernel_size=width,
        conv_layer_number=1,
        s4_states=s4_states,
        s4_seq_length=s4_seq_len,
        s4_l_width=width,
        s4_use_gating=True,
        s4_include_interblock=True,
        s4_use_circulant_slice=True,
        use_convolution=True,
        bidirectional=True,
        use_layer_norm_mixer=False,
        use_symmetric_conv=False,
        shift_project=False,
        partial_evaluation=True,
    )


def _compute_kernels(op, vstate, token_size: int):
    sigma = jnp.asarray(vstate.samples)
    if sigma.ndim == 3:
        sigma = sigma.reshape(-1, sigma.shape[-1])

    eta_win, mels, lengths, sites = op.operator.apply_operator(sigma)
    idx = get_idx(jnp.cumsum(lengths))
    centers = (idx, sites)

    fast_kernel = TurboTurboOperator._get_kernel(vstate, op)
    fast_vals = fast_kernel(None, vstate.parameters, sigma, (eta_win, mels, centers))

    eta_full = _embed_windows_into_full(
        sigma,
        eta_win,
        lengths,
        sites,
        width=op._width,
        token_size=token_size,
    )

    params_tree = vstate.parameters
    try:
        has_params_key = "params" in params_tree
    except TypeError:
        has_params_key = False
    std_params = params_tree["params"] if has_params_key else params_tree

    standard_kernel = TurboTurboOperator._get__standard_kernel(vstate, op)
    standard_vals = standard_kernel(None, std_params, sigma, (eta_full, mels))

    return (
        np.array(fast_vals),
        np.array(standard_vals),
        np.array(sigma),
        np.array(eta_full),
        np.array(mels),
    )


def _report(
    *,
    L: int,
    width: int,
    token_size: int,
    embedding_dim: int,
    hidden_dim: int,
    n_blocks: int,
    s4_states: int,
    fast_vals: np.ndarray,
    standard_vals: np.ndarray,
):
    abs_err = np.max(np.abs(fast_vals - standard_vals))
    denom = np.max(np.abs(standard_vals))
    rel_err = abs_err / denom if denom > 0 else 0.0
    print(
        (
            "CASE "
            f"L={L} width={width} token_size={token_size} "
            f"embedding_dim={embedding_dim} hidden_dim={hidden_dim} "
            f"n_blocks={n_blocks} s4_states={s4_states}"
        ),
        file=sys.__stdout__,
    )
    print(
        (
            f"values dtype: fast={fast_vals.dtype}, standard={standard_vals.dtype} "
            f"shape={fast_vals.shape}"
        ),
        file=sys.__stdout__,
    )
    print(
        f"max abs err: {abs_err:.3e}, max rel err: {rel_err:.3e}",
        file=sys.__stdout__,
    )


@pytest.mark.parametrize(
    "case",
    LOCAL_UPDATE_CASES_TO_RUN,
)
def test_local_update_kernel_matches_standard_tfim(case):
    L = case["L"]
    width = case["width"]
    token_size = case["token_size"]
    assert L % token_size == 0, "token_size must divide system size"

    model = _build_test_model(
        L,
        token_size,
        width,
        embedding_dim=case["embedding_dim"],
        hidden_dim=case["hidden_dim"],
        n_blocks=case["n_blocks"],
        s4_states=case["s4_states"],
    )
    system = setup_system_long_range_ising(
        N=L,
        J=1.0,
        alpha=3.0,
        width=width,
        token_size=token_size,
        fast=True,
        fast_sampler=False,
        n_chains=4,
        sweep_size=L,
    )
    system.setup_sampling(n_samples=8, n_discard_per_chain=0)
    vstate = system.get_variational_state(
        model,
        seed=case["seed"],
        sampler_seed=case["seed"],
    )

    # Populate samples before evaluating kernels
    _ = np.array(vstate.samples)

    fast_vals, standard_vals, _, _, _ = _compute_kernels(system.H, vstate, token_size)
    _report(
        L=L,
        width=width,
        token_size=token_size,
        embedding_dim=case["embedding_dim"],
        hidden_dim=case["hidden_dim"],
        n_blocks=case["n_blocks"],
        s4_states=case["s4_states"],
        fast_vals=fast_vals,
        standard_vals=standard_vals,
    )
    np.testing.assert_allclose(
        fast_vals,
        standard_vals,
        rtol=1e-5,
        atol=1e-4,
    )


if __name__ == "__main__":
    for entry in LOCAL_UPDATE_CASES:
        test_local_update_kernel_matches_standard_tfim(entry)
