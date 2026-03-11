import os

# Ensure deterministic, CPU-only execution for JAX/netket in tests
os.environ.setdefault("JAX_ENABLE_X64", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import sys

import dysonnet.DysonNQS as mnqs
import pytest
from dysonnet.custom_operator import (
    TurboTurboOperator,
    _embed_windows_into_full,
    get_idx,
)
from dysonnet.utils import setup_system_j1j2, setup_system_long_range_ising

FAST_MODE = os.environ.get("PYTEST_FAST") == "1"

J1J2_CASES = [
    (12, 1, 2),   # baseline small system, single-spin tokens
    (12, 2, 2),   # matches the original test
    (20, 1, 3),   # requested larger system size
    (20, 2, 2),   # larger system with token grouping
    (40, 2, 2),   # coarse tokens; window spans enough neighbours
    (80, 2, 2),   # coarse tokens; window spans enough neighbours
]
J1J2_FAST_CASES = [
    (12, 2, 2),
    (40, 2, 2),
]
J1J2_CASES_TO_RUN = J1J2_FAST_CASES if FAST_MODE else J1J2_CASES


def _build_test_model(L: int, token_size: int, width: int, *, shift_project: bool = False):
    """Create a tiny gated S4 model that supports partial evaluation."""
    s4_seq_len = L // token_size
    return mnqs.DysonNetFactory.create_model(
        token_size=token_size,
        embedding_dim=8,
        hidden_dim=8,
        n_blocks=1,
        use_complex=True,
        use_logcosh_output=True,
        logcosh_hidden=2,
        conv_kernel_size=width,
        conv_layer_number=1,
        s4_states=4,
        s4_seq_length=s4_seq_len,
        s4_l_width=width,
        s4_use_gating=True,
        s4_include_interblock=True,
        s4_use_circulant_slice=True,
        use_convolution=True,
        bidirectional=True,
        use_layer_norm_mixer=False,
        use_symmetric_conv=False,
        shift_project=shift_project,
        partial_evaluation=True,
    )


def _compute_kernels(op: TurboTurboOperator, vstate, token_size: int):
    """Evaluate both custom and standard kernels on the same samples."""
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

    return np.array(fast_vals), np.array(standard_vals)


def _report_errors(fast_vals: np.ndarray, standard_vals: np.ndarray) -> None:
    abs_err = np.max(np.abs(fast_vals - standard_vals))
    denom = np.max(np.abs(standard_vals))
    rel_err = abs_err / denom if denom > 0 else 0.0
    print(f"max abs err: {abs_err:.3e}, max rel err: {rel_err:.3e}", file=sys.__stdout__)


def test_kernel_matches_standard_tfim():
    L = 8
    width = 2
    token_size = 2

    model = _build_test_model(L, token_size, width)
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
    vstate = system.get_variational_state(model, seed=0, sampler_seed=0)

    # Populate samples before evaluating kernels
    _ = np.array(vstate.samples)

    fast_vals, standard_vals = _compute_kernels(system.H, vstate, token_size)
    _report_errors(fast_vals, standard_vals)
    np.testing.assert_allclose(
        fast_vals,
        standard_vals,
        rtol=1e-6,
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "L,token_size,width",
    J1J2_CASES_TO_RUN,
)
def test_kernel_matches_standard_j1j2(L, token_size, width):
    assert L % token_size == 0, "token_size must divide system size"

    model = _build_test_model(L, token_size, width)
    system = setup_system_j1j2(
        N=L,
        J1=1.0,
        J2=0.2,
        width=width,
        token_size=token_size,
        fast=True,
        fast_sampler=False,
        n_chains=4,
        sweep_size=L,
    )
    system.setup_sampling(n_samples=8, n_discard_per_chain=0)
    vstate = system.get_variational_state(model, seed=1, sampler_seed=1)

    _ = np.array(vstate.samples)

    fast_vals, standard_vals = _compute_kernels(system.H, vstate, token_size)
    _report_errors(fast_vals, standard_vals)
    np.testing.assert_allclose(
        fast_vals,
        standard_vals,
        rtol=1e-6,
        atol=1e-4,
    )


def test_kernel_matches_standard_j1j2_shift_project():
    L = 12
    token_size = 2
    width = 2

    model = _build_test_model(L, token_size, width, shift_project=True)
    system = setup_system_j1j2(
        N=L,
        J1=1.0,
        J2=0.2,
        width=width,
        token_size=token_size,
        fast=True,
        fast_sampler=False,
        n_chains=4,
        sweep_size=L,
    )
    system.setup_sampling(n_samples=8, n_discard_per_chain=0)
    vstate = system.get_variational_state(model, seed=2, sampler_seed=2)

    _ = np.array(vstate.samples)

    fast_vals, standard_vals = _compute_kernels(system.H, vstate, token_size)
    _report_errors(fast_vals, standard_vals)
    np.testing.assert_allclose(
        fast_vals,
        standard_vals,
        rtol=1e-6,
        atol=1e-4,
    )


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", False)
    test_kernel_matches_standard_tfim()
    test_kernel_matches_standard_j1j2()
