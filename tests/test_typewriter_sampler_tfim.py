import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from netket.utils import wrap_afun

import dysonnet.custom_sampler as cs
import dysonnet.DysonNQS as mnqs
from dysonnet.utils import setup_system_long_range_ising


jax.config.update("jax_enable_x64", False)

FAST_MODE = os.environ.get("PYTEST_FAST") == "1"

TYPEWRITER_CASES = [
    dict(L=12, width=2, token_size=1),
    dict(L=24, width=2, token_size=2),
    dict(L=48, width=2, token_size=4),
]
FAST_TYPEWRITER_CASES = [TYPEWRITER_CASES[0]]
TYPEWRITER_CASES_TO_RUN = (
    FAST_TYPEWRITER_CASES if FAST_MODE else TYPEWRITER_CASES
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
        s4_include_interblock=False,
        s4_use_circulant_slice=True,
        use_convolution=True,
        bidirectional=True,
        use_layer_norm_mixer=False,
        use_symmetric_conv=False,
        shift_project=False,
        partial_evaluation=True,
    )


def _burn_in_lut(sampler, machine, params, state, *, sweeps: int) -> tuple:
    for _ in range(sweeps):
        state, _ = sampler._sample_typewriter_sweep(machine, params, state)
    return state


def _compute_logp_deviation(sampler, params, state_before, debug):
    centers = jnp.asarray(debug["proposal_centers"])
    deltas = jnp.asarray(debug["replay_vectors"])
    accepted = jnp.asarray(debug["proposal_accepted"]).astype(jnp.bool_)

    interval_count, _ = centers.shape

    vector_cache = sampler.apply_full_headless(params, state_before.σ)
    vector = vector_cache[0] if isinstance(vector_cache, tuple) else vector_cache

    sigma = state_before.σ
    current_vector = vector
    abs_diffs = []

    for i in range(interval_count):
        centers_i = centers[i]
        sigma_prop = sampler.local_move_rule.apply(
            sigma, centers_i, jax.random.PRNGKey(0)
        )
        approx_logp = (
            sampler.machine_pow
            * cs.logcosh_activation(current_vector + deltas[i]).real
        )
        exact_logp = sampler.machine_pow * sampler.apply_full(params, sigma_prop).real
        abs_diffs.append(jnp.abs(exact_logp - approx_logp))

        accept_mask = accepted[i]
        sigma = jnp.where(accept_mask[:, None], sigma_prop, sigma)
        current_vector = jnp.where(
            accept_mask[:, None], current_vector + deltas[i], current_vector
        )

    abs_diffs = jnp.stack(abs_diffs, axis=0)
    return {
        "max_abs_logp": float(jnp.max(abs_diffs)),
        "mean_abs_logp": float(jnp.mean(abs_diffs)),
    }


def _run_typewriter_debug_case(case: dict, lut_multiply: float, *, seed: int):
    L = case["L"]
    width = case["width"]
    token_size = case["token_size"]

    model = _build_test_model(
        L,
        token_size,
        width,
        embedding_dim=8,
        hidden_dim=8,
        n_blocks=1,
        s4_states=4,
    )

    system = setup_system_long_range_ising(
        N=L,
        J=1.0,
        alpha=3.0,
        width=width,
        model=model,
        token_size=token_size,
        fast=True,
        fast_sampler=True,
        n_chains=8,
        sweep_size=L,
        simple_typewriter=False,
        dyn_tol_enabled=True,
        lut_multiply=lut_multiply,
    )
    system.setup_sampling(n_samples=8, n_discard_per_chain=0)

    vstate = system.get_variational_state(
        model,
        seed=seed,
        sampler_seed=seed,
    )
    # Force parameter initialization and sampler warmup inside NetKet.
    _ = np.asarray(vstate.samples)
    params = vstate.variables

    sampler = vstate.sampler
    assert isinstance(sampler, cs.MetropolisSampler)
    assert sampler.local_move_rule.move_kind == "single_flip"
    assert sampler.lut_multiply == pytest.approx(lut_multiply)

    machine = wrap_afun(model)
    state = sampler.reset(model, params, vstate.sampler_state)

    # Burn-in to equilibrate the LUT values.
    state = _burn_in_lut(sampler, machine, params, state, sweeps=12)

    state_before = state
    state, (_, _), debug = sampler._sample_typewriter(
        machine, params, state, True
    )

    errors = int(np.sum(np.asarray(debug["debug_wrong"])))
    total_error = float(np.sum(np.asarray(debug["debug_total_error"])))
    debug_errors = np.asarray(debug["debug_errors"])
    max_metropolis_error = float(np.max(debug_errors))
    mean_metropolis_error = float(np.mean(debug_errors))
    lut_updates = int(np.asarray(state.lut_update_count))
    freeze_iter = np.asarray(debug["freeze_iter"])
    interval_count = debug["proposal_centers"].shape[0]
    screened = freeze_iter < interval_count
    screened_count = int(np.sum(screened))
    screened_frac = float(np.mean(screened))
    mean_freeze_iter = float(np.mean(freeze_iter))

    logp_stats = _compute_logp_deviation(sampler, params, state_before, debug)

    return {
        "errors": errors,
        "total_error": total_error,
        "max_metropolis_error": max_metropolis_error,
        "mean_metropolis_error": mean_metropolis_error,
        "lut_updates": lut_updates,
        "screened_count": screened_count,
        "screened_frac": screened_frac,
        "mean_freeze_iter": mean_freeze_iter,
        **logp_stats,
    }


@pytest.mark.parametrize(
    "case",
    TYPEWRITER_CASES_TO_RUN,
)
def test_typewriter_sampler_tfim_debug_lut_multiply(case):
    print(
        (
            "TFIM typewriter sampler test: "
            "checking debug mismatches + log-amplitude deviation after LUT burn-in. "
            "Pass condition: errors == 0 for lut_multiply=1.5."
        ),
        file=sys.__stdout__,
    )
    print(
        (
            f"CASE L={case['L']} token_size={case['token_size']} "
            f"width={case['width']}"
        ),
        file=sys.__stdout__,
    )
    lut_values = [1.5] if FAST_MODE else [0.5, 1.0, 1.5, 2.0]
    results = {}

    for lut_multiply in lut_values:
        stats = _run_typewriter_debug_case(case, lut_multiply, seed=0)
        results[lut_multiply] = stats
        print(
            (
                "TFIM typewriter debug: "
                f"lut_multiply={lut_multiply} "
                f"errors={stats['errors']} "
                f"total_error={stats['total_error']:.3e} "
                f"metropolis_err[max={stats['max_metropolis_error']:.3e}, "
                f"mean={stats['mean_metropolis_error']:.3e}] "
                f"logp_err[max={stats['max_abs_logp']:.3e}, "
                f"mean={stats['mean_abs_logp']:.3e}] "
                f"screened={stats['screened_count']} "
                f"screened_frac={stats['screened_frac']:.2f} "
                f"mean_freeze_iter={stats['mean_freeze_iter']:.2f} "
                f"lut_updates={stats['lut_updates']}"
            ),
            file=sys.__stdout__,
        )

    # Default lut_multiply should produce no mismatches after burn-in.
    assert results[1.5]["errors"] == 0
