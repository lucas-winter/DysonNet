import os
os.environ["JAX_ENABLE_X64"] = "0"   # must be set before importing jax
os.environ["JAX_PLATFORMS"] = "cpu"  # optional: to force CPU if you’re on Metal/TPU

import jax
jax.config.update("jax_enable_x64", False)

import netket as nk


import itertools

import jax.numpy as jnp
import numpy as np

from dysonnet.custom_operator import (
    _embed_windows_into_full,
    apply_operator_power,
)


class DummyFlipOperator:
    """
    Simple operator that flips every individual spin.
    Each configuration of length L produces L children with
    matrix elements equal to (site_index + 1).
    """

    def apply_operator(self, sigma: jnp.ndarray):
        sigma = jnp.asarray(sigma)
        batch, length = sigma.shape

        lengths = length * jnp.ones((batch,), dtype=jnp.int32)
        parent_idx = jnp.repeat(jnp.arange(batch, dtype=jnp.int32), length)
        sites = jnp.tile(jnp.arange(length, dtype=jnp.int32), batch)

        flipped = -sigma[parent_idx, sites]
        eta_win = flipped[:, None]  # width=0 ⇒ single-spin window
        mels = (sites + 1).astype(sigma.dtype)

        return eta_win, mels, lengths, sites


def test_embed_windows():
    sigma_full = jnp.array(
        [
            [1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1],
        ],
        dtype=jnp.int32,
    )
    lengths = jnp.array([2, 1], dtype=jnp.int32)
    sites = jnp.array([1, 3, 4], dtype=jnp.int32)
    eta_win = jnp.array(
        [
            [-2, -3, -4],
            [5, 6, 7],
            [8, 9, 10],
        ],
        dtype=jnp.int32,
    )

    eta_full = _embed_windows_into_full(
        sigma_full,
        eta_win,
        lengths,
        sites,
        width=1,
        token_size=1,
    )

    expected = jnp.array(
        [
            [-2, -3, -4, 1, 1, 1],
            [1, 1, 5, 6, 7, 1],
            [-1, -1, -1, 8, 9, 10],
        ],
        dtype=jnp.int32,
    )

    np.testing.assert_array_equal(
        np.array(eta_full),
        np.array(expected),
        err_msg="Window embedding should overwrite the correct sites.",
    )


def test_apply_operator_power_shapes_and_mels():
    sigma = jnp.array([[1, 1, 1]], dtype=jnp.int32)
    power = 2
    operator = DummyFlipOperator()

    eta_full, mels, lengths = apply_operator_power(
        operator,
        sigma,
        power,
        width=0,
        token_size=1,
    )

    expected_child_count = sigma.shape[-1] ** power
    assert (
        eta_full.shape == (expected_child_count, sigma.shape[-1])
    ), "Full configs shape should match total generated states."
    assert (
        mels.shape == (expected_child_count,)
    ), "Matrix elements should align with generated states."
    np.testing.assert_array_equal(
        np.array(lengths),
        np.array([expected_child_count]),
        err_msg="Lengths should count children per original configuration.",
    )

    combos = [np.prod([site + 1 for site in seq]) for seq in itertools.product(range(sigma.shape[-1]), repeat=power)]
    np.testing.assert_array_equal(
        np.array(mels),
        np.array(combos),
        err_msg="Matrix elements should be the product over each path.",
    )

    base = np.array(sigma)
    expected_states = []
    for sequence in itertools.product(range(base.shape[-1]), repeat=power):
        conf = base[0].copy()
        for site in sequence:
            conf[site] *= -1
        expected_states.append(conf)
    np.testing.assert_array_equal(
        np.array(eta_full),
        np.array(expected_states),
        err_msg="Full configurations should reflect sequential spin flips.",
    )


if __name__ == "__main__":
    test_embed_windows()
    test_apply_operator_power_shapes_and_mels()
    print("All custom operator tests passed.")
