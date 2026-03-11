# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Any, NamedTuple
from collections.abc import Callable
from dataclasses import dataclass
from textwrap import dedent
from dysonnet.partial_evaluation import slice_tokens_jit
import netket as nk

import numpy as np

import jax
from flax import linen as nn
from jax import numpy as jnp
from tqdm import tqdm
from netket.hilbert import AbstractHilbert

from netket.utils import mpi, wrap_afun
from netket.utils.types import PyTree, DType

from netket.utils.deprecation import warn_deprecation
from netket.utils import struct

from netket.jax.sharding import (
    device_count,
    shard_along_axis,
)
from netket.jax import apply_chunked, dtype_real

from netket.sampler.base import Sampler, SamplerState
from netket.sampler.rules import MetropolisRule

import matplotlib.pyplot as plt

def logcosh_activation(x: jnp.ndarray) -> jnp.ndarray:
    """Log-cosh activation function."""
    activation = jnp.logaddexp(x, -x) - jnp.log(2)
    return jnp.sum(activation, axis=-1)

def get_centers(
    eta: jnp.ndarray, 
    sigma: jnp.ndarray
) -> jnp.ndarray:
    """
    For each row in eta vs sigma:
      - If no entry differs, return -1.
      - Otherwise, compute the mean of all changed-site indices and floor it.

    Args:
      eta:   shape (batch, N), proposed configurations
      sigma: shape (batch, N), current configurations

    Returns:
      centers: shape (batch,), int32; ⟂mean(indexes_of_changes) or −1 if no change.
    """
    # mask of where they differ
    diff = eta != sigma                  # (batch, N), bool

    # how many sites changed in each sample
    counts = jnp.sum(diff, axis=1)       # (batch,), int

    # prepare indices [0,1,2,...,N-1]
    N = eta.shape[1]
    idx = jnp.arange(N, dtype=jnp.int32) # (N,)

    # sum of the changed-site indices
    sum_idx = jnp.sum(diff * idx[None, :], axis=1)  # (batch,)

    # avoid division by zero by masking counts==0 → use 1 instead
    safe_counts = jnp.where(counts > 0, counts, 1)

    # integer division gives us floor(mean)
    means = sum_idx // safe_counts      # (batch,)

    # if counts==0, overwrite with −1
    return jnp.where(counts > 0, means, -1)



def _resolve_schedule_value(
    base: int,
    explicit: int | None,
    multiplier: float | None,
) -> float:
    """
    Resolve a sweep target either from an explicit integer or from a multiplier
    applied to the base sweep size.
    """
    if explicit is not None:
        return float(explicit)
    if multiplier is not None:
        return float(base) * float(multiplier)
    return float(base)


@dataclass
class SweepRampConfig:
    """
    Configuration for a piecewise-linear sweep-size ramp.

    Parameters
    ----------
    plateau_iters :
        Number of optimisation iterations during which the sweep size is held
        constant at the starting value.
    ramp_iters :
        Number of iterations over which the sweep size is linearly interpolated
        from the starting value to the final value.
    start_sweeps :
        Absolute number of sweeps to use during the plateau. If omitted, the
        sampler's baseline sweep size is used.
    end_sweeps :
        Absolute number of sweeps to use after the ramp finishes. If omitted,
        falls back to `end_multiplier` or the baseline sweep size.
    start_multiplier :
        Multiplier applied to the baseline sweep size to obtain the plateau
        value when `start_sweeps` is not provided.
    end_multiplier :
        Multiplier applied to the baseline sweep size to obtain the final value
        when `end_sweeps` is not provided.
    min_sweeps :
        Lower bound applied to the computed sweep size.
    max_sweeps :
        Optional upper bound applied to the computed sweep size.
    """

    plateau_iters: int = 0
    ramp_iters: int = 1
    start_sweeps: int | None = None
    end_sweeps: int | None = None
    start_multiplier: float | None = None
    end_multiplier: float | None = None
    min_sweeps: int = 1
    max_sweeps: int | None = None

    def build(self, base_sweep: int) -> Callable[[int], int]:
        """
        Create a schedule callable from the configuration.
        """
        if base_sweep <= 0:
            raise ValueError("Base sweep size must be positive.")

        start_value = _resolve_schedule_value(
            base_sweep, self.start_sweeps, self.start_multiplier
        )
        end_value = _resolve_schedule_value(
            base_sweep, self.end_sweeps, self.end_multiplier
        )

        min_sweeps = max(int(self.min_sweeps), 1)
        max_sweeps = None
        if self.max_sweeps is not None:
            max_sweeps = max(int(self.max_sweeps), min_sweeps)

        plateau_iters = max(int(self.plateau_iters), 0)
        ramp_iters = max(int(self.ramp_iters), 0)

        ramp_end = plateau_iters + ramp_iters

        def schedule(iteration: int) -> int:
            it = max(int(iteration), 0)

            if it < plateau_iters:
                value = start_value
            elif ramp_iters == 0 or it >= ramp_end:
                value = end_value
            else:
                progress = (it - plateau_iters) / ramp_iters
                value = start_value + (end_value - start_value) * progress

            if max_sweeps is not None:
                value = min(value, max_sweeps)

            value = max(value, min_sweeps)
            return int(round(value))

        return schedule


def check_not_collides(prev_centers: jnp.ndarray,
                 new_center: jnp.ndarray,
                 collision_width: int, L : int) -> jnp.ndarray:
    """
    prev_centers: shape (t, batch), dtype int, with -1 meaning “no previous center”
    new_center:  shape (batch,), dtype int
    returns:     shape (t, batch) boolean mask: True = no collision, False = collision
    """
    # 1) Mask out all the “empty” entries so they never collide:
    empty = prev_centers < 0             # True wherever prev_centers == -1

    # 2) For the rest, compute the normal distance test:
    #    (broadcast new_center up to shape (t, batch))
    dist = jnp.abs(prev_centers - new_center[None, :])
    dist = jnp.minimum(dist, L - dist)
    
    no_conflict = dist > collision_width

    # 3) Combine: empty slots are always “no collision”:
    return empty | no_conflict


@struct.dataclass
class LocalMoveRule:
    """
    Encodes a local update σ -> σ' around centers.

    `radius` is the maximum distance (in sites) from the central index that this
    rule may modify. For single-spin flips, radius = 0.
    """

    radius: int = 0
    move_kind: str = "single_flip"

    def apply(self, σ: jnp.ndarray, centers: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


@struct.dataclass
class SingleFlipRule(LocalMoveRule):
    radius: int = 0
    move_kind: str = "single_flip"

    def apply(self, σ: jnp.ndarray, centers: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        batch = σ.shape[0]
        idx = jnp.arange(batch, dtype=jnp.int32)
        return σ.at[idx, centers].mul(-1)


@struct.dataclass
class FlipFlopRule(LocalMoveRule):
    """
    Flip–flop on bond (center, center + delta). Acts only when spins are
    anti-parallel.
    """

    delta: int = 1
    move_kind: str = "flip_flop"
    radius: int = struct.field(pytree_node=False, default=0)

    def __post_init__(self):
        object.__setattr__(self, "radius", abs(int(self.delta)))

    def apply(self, σ: jnp.ndarray, centers: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        batch, L = σ.shape
        idx = jnp.arange(batch, dtype=jnp.int32)
        j = (centers + self.delta) % L

        s_i = σ[idx, centers]
        s_j = σ[idx, j]
        mask = s_i != s_j

        new_i = jnp.where(mask, -s_i, s_i)
        new_j = jnp.where(mask, -s_j, s_j)

        σ_new = σ
        σ_new = σ_new.at[idx, centers].set(new_i)
        σ_new = σ_new.at[idx, j].set(new_j)
        return σ_new


@struct.dataclass
class CompositeMoveRule(LocalMoveRule):
    """
    A move that randomly chooses one of several sub-rules, with given probabilities.
    """

    rules: tuple = struct.field(pytree_node=True, default_factory=tuple)
    probs: jnp.ndarray | None = struct.field(pytree_node=True, default=None)
    move_kind: str = "composite"
    radius: int = struct.field(pytree_node=False, default=0)

    def __post_init__(self):
        if self.probs is None:
            raise ValueError("CompositeMoveRule: probs must be provided.")
        if self.probs.ndim != 1 or self.probs.shape[0] != len(self.rules):
            raise ValueError("CompositeMoveRule: probs must be 1D with length len(rules).")
        total = jnp.sum(self.probs)
        if total <= 0:
            raise ValueError("CompositeMoveRule: probs must sum to a positive value.")
        object.__setattr__(self, "probs", self.probs / total)
        object.__setattr__(self, "radius", max(int(r.radius) for r in self.rules))

    def apply(self, σ: jnp.ndarray, centers: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        key_rule, key_move = jax.random.split(key)
        log_probs = jnp.log(self.probs)
        rule_idx = jax.random.categorical(key_rule, log_probs)

        def _make_branch(i):
            def branch(args):
                σ_in, centers_in, key_in = args
                return self.rules[i].apply(σ_in, centers_in, key_in)
            return branch

        branches = tuple(_make_branch(i) for i in range(len(self.rules)))
        return jax.lax.switch(rule_idx, branches, (σ, centers, key_move))

class MetropolisSamplerState(SamplerState):
    """
    State for a Metropolis sampler.

    Contains the current configuration, the RNG state and the (optional)
    state of the transition rule.
    """

    σ: jnp.ndarray = struct.field(
        sharded=struct.ShardedFieldSpec(
            sharded=True, deserialization_function="relaxed-ignore-errors"
        )
    )
    """Current batch of configurations in the Markov chain."""
    log_prob: jnp.ndarray = struct.field(sharded=True, serialize=False)
    """Log probabilities of the current batch of configurations σ in the Markov chain."""
    rng: jnp.ndarray = struct.field(
        sharded=struct.ShardedFieldSpec(
            sharded=True, deserialization_function="relaxed-rng-key"
        )
    )
    """State of the random number generator (key, in jax terms)."""
    rule_state: Any | None
    """Optional state of the transition rule."""

    n_steps_proc: int = struct.field(default_factory=lambda: jnp.zeros((), dtype=int))
    """Number of moves performed along the chains in this process since the last reset."""
    n_accepted_proc: jnp.ndarray = struct.field(
        sharded=struct.ShardedFieldSpec(
            sharded=True, deserialization_function="relaxed-ignore-errors"
        )
    )
    """Number of accepted transitions among the chains in this process since the last reset."""

    """Accumulated output‐vectors (for fast sampling)."""
    interval_index : jnp.ndarray = struct.field(
        sharded=struct.ShardedFieldSpec(
            sharded=True, deserialization_function="relaxed-ignore-errors"
        )
    )
    freeze_correction: jnp.ndarray = struct.field(
        pytree_node=True, default_factory=lambda: jnp.asarray(1.0, dtype=jnp.float32)
    )

    lut_values: jnp.ndarray | None = struct.field(pytree_node=True, default=None)
    lut_counts: jnp.ndarray | None = struct.field(pytree_node=True, default=None)
    lut_last_update_step: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.asarray(0, dtype=jnp.int32)
    )
    lut_update_count: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.asarray(0, dtype=jnp.int32)
    )
    params_version: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.asarray(0, dtype=jnp.int32)
    )
    typewriter_target_steps: jnp.ndarray = struct.field(
        pytree_node=True, default_factory=lambda: jnp.asarray(0, dtype=jnp.int32)
    )
    iteration: jnp.ndarray = struct.field(
        pytree_node=True, default_factory=lambda: jnp.asarray(0, dtype=jnp.int32)
    )


    def __init__(
        self,
        σ: jnp.ndarray,
        rng: jnp.ndarray,
        rule_state: Any | None,
        log_prob: jnp.ndarray | None = None,
    ):
        self.σ = σ
        self.rng = rng
        self.rule_state = rule_state

        if log_prob is None:
            log_prob = jnp.full(self.σ.shape[:-1], -jnp.inf, dtype=float)
        self.log_prob = shard_along_axis(log_prob, axis=0)
        self.interval_index = shard_along_axis(jnp.zeros_like(log_prob, dtype=jnp.int32), axis=0)
        self.freeze_correction = jnp.asarray(1.0, dtype=jnp.float32)

        self.n_accepted_proc = shard_along_axis(
            jnp.zeros(σ.shape[0], dtype=int), axis=0
        )
        self.n_steps_proc = jnp.zeros((), dtype=int)
        self.lut_values = None
        self.lut_counts = None
        self.lut_last_update_step = jnp.asarray(0, dtype=jnp.int32)
        self.lut_update_count = jnp.asarray(0, dtype=jnp.int32)
        self.params_version = jnp.asarray(0, dtype=jnp.int32)
        self.typewriter_target_steps = jnp.asarray(0, dtype=jnp.int32)
        self.iteration = jnp.asarray(0, dtype=jnp.int32)
        super().__init__()

    @property
    def acceptance(self) -> float | None:
        """The fraction of accepted moves across all chains and MPI processes.

        The rate is computed since the last reset of the sampler.
        Will return None if no sampling has been performed since then.
        """
        if self.n_steps == 0:
            return None

        return self.n_accepted / self.n_steps

    @property
    def n_steps(self) -> int:
        """Total number of moves performed across all processes since the last reset."""
        return self.n_steps_proc * mpi.n_nodes

    @property
    def n_accepted(self) -> int:
        """Total number of moves accepted across all processes since the last reset."""
        # jit sum for gda
        res, _ = mpi.mpi_sum_jax(jax.jit(jnp.sum)(self.n_accepted_proc))
        return res

    def __repr__(self):
        try:
            if self.n_steps > 0:
                acc_string = f"# accepted = {self.n_accepted}/{self.n_steps} ({self.acceptance * 100}%), "
            else:
                acc_string = ""

            return f"{type(self).__name__}({acc_string}rng state={self.rng})"
        except TypeError:
            return f"{type(self).__name__}(???, rng state={self.rng})"

    def __process_deserialization_updates__(self, updates):
        # In netket 3.15 we changed the default dtype of samples
        # to integer dtypes in most of the time. Without this,
        # deserialization of old files would be broken.
        if self.σ.dtype != updates["σ"].dtype:
            updates["σ"] = updates["σ"].astype(self.σ.dtype)
        return updates


def _assert_good_sample_shape(samples, shape, dtype, obj=""):
    canonical_dtype = jax.dtypes.canonicalize_dtype(dtype)
    if samples.shape != shape or samples.dtype != canonical_dtype:
        raise ValueError(
            dedent(
                f"""

            The samples returned by the {obj} have `shape={samples.shape}` and
            `dtype={samples.dtype}`, but the sampler requires `shape={shape} and
            `dtype={canonical_dtype}` (canonicalized from {dtype}).

            If you are using a custom transition rule, check that it returns the
            correct shape and dtype.

            If you are using a built-in transition rule, there might be a mismatch
            between hilbert spaces, or it's a bug in NetKet.

            """
            )
        )


def _assert_good_log_prob_shape(log_prob, n_chains_per_rank, machine):
    if log_prob.shape != (n_chains_per_rank,):
        raise ValueError(
            dedent(
                f"""

            The output of the model {machine} has `shape={log_prob.shape}`, but
            `shape=({n_chains_per_rank},)` was expected.

            This might be because of an hilbert space mismatch or because your
            model is ill-configured.

            """
            )
        )


def _round_n_chains_to_next_multiple(
    n_chains, n_chains_per_whatever, n_whatever, whatever_str
):
    # small helper function to round the number of chains to the next multiple of [whatever]
    # here [whatever] can be e.g. mpi ranks or jax devices
    # if n_chains is None and n_chains_per_whatever is None:
    #    n_chains_per_whatever = default
    if n_chains is not None and n_chains_per_whatever is not None:
        raise ValueError(
            f"Cannot specify both `n_chains` and `n_chains_per_{whatever_str}`"
        )
    elif n_chains is not None:
        n_chains_per_whatever = max(int(np.ceil(n_chains / n_whatever)), 1)
        if n_chains_per_whatever * n_whatever != n_chains:
            if mpi.rank == 0:
                import warnings

                warnings.warn(
                    f"Using {n_chains_per_whatever} chains per {whatever_str} among {n_whatever} {whatever_str}s "
                    f"(total={n_chains_per_whatever * n_whatever} instead of n_chains={n_chains}). "
                    f"To directly control the number of chains on every {whatever_str}, specify "
                    f"`n_chains_per_{whatever_str}` when constructing the sampler. "
                    f"To silence this warning, either use `n_chains_per_{whatever_str}` or use `n_chains` "
                    f"that is a multiple of the number of {whatever_str}s",
                    category=UserWarning,
                    stacklevel=2,
                )
    return n_chains_per_whatever * n_whatever





class MetropolisSampler(Sampler):
    r"""
    Metropolis-Hastings sampler for a Hilbert space according to a specific transition rule.

    The transition rule is used to generate a proposed state :math:`s^\prime`, starting from the
    current state :math:`s`. The move is accepted with probability

    .. math::

        A(s \rightarrow s^\prime) = \mathrm{min} \left( 1,\frac{P(s^\prime)}{P(s)} e^{L(s,s^\prime)} \right) ,

    where the probability being sampled from is :math:`P(s)=|M(s)|^p`. Here :math:`M(s)` is a
    user-provided function (the machine), :math:`p` is also user-provided with default value :math:`p=2`,
    and :math:`L(s,s^\prime)` is a suitable correcting factor computed by the transition kernel.

    The dtype of the sampled states can be chosen.
    """

    rule: MetropolisRule = None  # type: ignore
    """The Metropolis transition rule."""
    sweep_size: int = struct.field(pytree_node=False, default=None)
    """Number of sweeps for each step along the chain. Defaults to the number
    of sites in the Hilbert space."""
    n_chains: int = struct.field(pytree_node=False)
    """Total number of independent chains across all MPI ranks and/or devices."""
    chunk_size: int | None = struct.field(pytree_node=False, default=None)
    """Chunk size for evaluating wave functions."""
    reset_chains: bool = struct.field(pytree_node=False, default=False)
    """If True, resets the chain state when `reset` is called on every new sampling."""
    reset_chain_length: int   = struct.field(pytree_node=False, default=5)
    width:             int   = struct.field(pytree_node=False, default=5)
    token_size:        int   = struct.field(pytree_node=False, default=1)
    shift_project:     bool  = struct.field(pytree_node=False, default=False)
    sz_sector:         int | None = struct.field(pytree_node=False, default=None)
    local_move_rule:   LocalMoveRule | None = struct.field(pytree_node=False, default=None)
    output_size:       int   = struct.field(pytree_node=False, default=1)
    sample_fast:       bool  = struct.field(pytree_node=False, default=False)
    accept_tol : float = struct.field(pytree_node=False, default=0.01)
    global_proposal : bool = struct.field(pytree_node=False, default=False)
    mode : str = struct.field(pytree_node=False, default="fast")
    simple_typewriter : bool = struct.field(pytree_node=False, default=False)
    interval_mult:       int   = struct.field(pytree_node=False, default=1)
    max_accept:        int   = struct.field(pytree_node=False, default=jnp.inf)
    dyn_tol_enabled: bool = struct.field(pytree_node=False, default=False)
    lut_max_flips: int | None = struct.field(pytree_node=False, default=None)
    lut_refresh_every: int = struct.field(pytree_node=False, default=50)
    lut_calib_sweeps: int = struct.field(pytree_node=False, default=2)
    lut_default: float = struct.field(pytree_node=False, default=1e-2)
    lut_decay: float = struct.field(pytree_node=False, default=0.1)
    lut_floor: float = struct.field(pytree_node=False, default=5e-3)
    lut_k_max: int = struct.field(pytree_node=False, default=0)
    lut_warmup_updates: int = struct.field(pytree_node=False, default=5)
    lut_multiply: float = struct.field(pytree_node=False, default=1.5)
    _stagger_sign: jnp.ndarray | None = struct.field(pytree_node=True, default=None)
    base_sweep_size: int = struct.field(pytree_node=False, default=0)
    sweep_schedule: Callable[[int], int] | None = struct.field(
        pytree_node=False, default=None
    )
    sweep_schedule_config: SweepRampConfig | None = struct.field(
        pytree_node=False, default=None
    )
    use_reduced_precision: bool = struct.field(pytree_node=False, default=False)

    #Headless apply functions   
    apply_full_headless : nk.utils.HashablePartial = struct.field(pytree_node=False, default=None)
    apply_full_headless_no_cache : nk.utils.HashablePartial = struct.field(pytree_node=False, default=None)
    apply_headless_partial : nk.utils.HashablePartial = struct.field(pytree_node=False, default=None)
    apply_partial : nk.utils.HashablePartial = struct.field(pytree_node=False, default=None)
    apply_full : nk.utils.HashablePartial = struct.field(pytree_node=False, default=None)

    def __init__(
        self,
        hilbert: AbstractHilbert,
        rule: MetropolisRule,
        *,
        n_sweeps: int = None,
        sweep_size: int = None,
        reset_chains: bool = False,
        n_chains: int | None = None,
        n_chains_per_rank: int | None = None,
        chunk_size: int | None = None,
        machine_pow: int = 2,
        dtype: DType = None,
        width: int = 5,
        token_size : int = 1,
        model: nn.Module | None = None,
        reset_chain_length: int = 5,
        output_size: int | None = None,
        sample_fast : bool = False,
        accept_tol : float = 0.01,
        mode = "fast",
        global_proposal: bool = False,
        simple_typewriter : bool = False, 
        interval_mult : int = 1, 
        max_accept : int = jnp.inf, 
        dyn_tol_enabled: bool = False,
        lut_max_flips: int | None = None,
        lut_refresh_every: int = 50,
        lut_calib_sweeps: int = 2,
        lut_default: float = 1e-2,
        use_reduced_precision: bool = False,
        lut_decay: float = 0.98,
        lut_floor: float = 1e-4,
        lut_multiply: float = 1.5,
        lut_warmup_updates: int = 5,
        stagger_pattern: jnp.ndarray | None = None,
        sweep_ramp: SweepRampConfig | Callable[[int], int] | None = None,
        sz_sector: int | None = None,
        local_move_rule: LocalMoveRule | None = None,
    ):
        """
        Constructs a Metropolis Sampler.

        Args:
            hilbert: The Hilbert space to sample.
            rule: A `MetropolisRule` to generate random transitions from a given state as
                well as uniform random states.
            n_chains: The total number of independent Markov chains across all MPI ranks.
                Either specify this or `n_chains_per_rank`. If MPI is disabled, the two are equivalent;
                if MPI is enabled and `n_chains` is specified, then every MPI rank will run
                `n_chains/mpi.n_nodes` chains. In general, we recommend specifying `n_chains_per_rank`
                as it is more portable.
            n_chains_per_rank: Number of independent chains on every MPI rank (default = 16).
                                If netket_experimental_sharding is enabled this is interpreted as the number
                                of independent chains on every jax device, and the n_chains_per_rank
                                property of the sampler will return the total number of chains on all devices.
            chunk_size: Chunk size for evaluating the ansatz while sampling. Must divide n_chains_per_rank.
            sweep_size: Number of sweeps for each step along the chain.
                This is equivalent to subsampling the Markov chain. (Defaults to the number of sites
                in the Hilbert space.)
            reset_chains: If True, resets the chain state when `reset` is called on every
                new sampling (default = False).
            machine_pow: The power to which the machine should be exponentiated to generate
                the pdf (default = 2).
            dtype: The dtype of the states sampled (default = jnp.float64).
            dyn_tol_enabled: Enable adaptive acceptance tolerance using a lookup table.
            lut_max_flips: Maximum number of accepted flips tracked in the LUT (defaults to sweep_size).
            lut_refresh_every: Number of sweeps between LUT calibration passes.
            lut_calib_sweeps: Number of synthetic sweeps used per calibration pass.
            lut_default: Fallback tolerance when no calibration data is available.
            lut_decay: Exponential decay applied to stored LUT maxima before inserting new values.
            lut_floor: Minimum per-bin tolerance returned by the LUT.
            lut_warmup_updates: Minimum number of LUT updates before enabling adaptive thresholds.
            stagger_pattern: Optional custom staggered magnetization pattern (defaults to alternating ±1).
            sweep_ramp: Optional schedule controlling the effective sweep size as a
                function of the optimisation iteration. Provide either a callable that
                maps the iteration index to a sweep size, or a `SweepRampConfig`.
        """

        # Validate the inputs
        if not isinstance(rule, MetropolisRule):
            raise TypeError(
                f"The second positional argument, rule, must be a MetropolisRule but "
                f"`type(rule)={type(rule)}`."
            )

        if not isinstance(reset_chains, bool):
            raise TypeError("reset_chains must be a boolean.")

        if n_sweeps is not None:
            warn_deprecation(
                "Specifying `n_sweeps` when constructing sampler is deprecated. Please use `sweep_size` instead."
            )
            if sweep_size is not None:
                raise ValueError("Cannot specify both `sweep_size` and `n_sweeps`")
            sweep_size = n_sweeps

        if sweep_size is None:
            sweep_size = hilbert.size

        # Default n_chains per rank, if unset
        if n_chains is None and n_chains_per_rank is None:
            # TODO set it to a few hundred if on GPU?
            n_chains_per_rank = 16

        n_chains = _round_n_chains_to_next_multiple(
            n_chains,
            n_chains_per_rank,
            device_count(),
            "rank",
        )
        n_chains_per_rank = n_chains // device_count()

        if chunk_size is not None and n_chains_per_rank % chunk_size != 0:
            raise ValueError(
                f"Chunk size must divide number of chains per rank, {n_chains_per_rank}"
            )
        self.chunk_size = chunk_size

        super().__init__(
            hilbert=hilbert,
            machine_pow=machine_pow,
            dtype=dtype,
        )

        self.n_chains = n_chains
        self.reset_chains = reset_chains
        self.rule = rule
        self.sweep_size = sweep_size
        self.base_sweep_size = int(sweep_size)
        self.sweep_schedule = None
        self.sweep_schedule_config = None
        self.reset_chain_length = reset_chain_length
        self.width = width
        self.output_size = output_size if output_size is not None else 1
        self.sample_fast = sample_fast
        self.token_size = token_size
        self.shift_project = bool(getattr(model, "shift_project", False))
        if local_move_rule is None:
            local_move_rule = SingleFlipRule()
        self.local_move_rule = local_move_rule
        self.accept_tol = accept_tol
        self.global_proposal = global_proposal
        self.simple_typewriter = simple_typewriter
        self.interval_mult = interval_mult 
        self.max_accept = max_accept
        self.dyn_tol_enabled = dyn_tol_enabled
        self.use_reduced_precision = use_reduced_precision
        if sz_sector is not None:
            sz_sector = int(sz_sector)
            if abs(sz_sector) > self.hilbert.size:
                raise ValueError(
                    f"Requested sz_sector={sz_sector} exceeds system size {self.hilbert.size}"
                )
        self.sz_sector = sz_sector
        self.lut_refresh_every = lut_refresh_every
        self.lut_calib_sweeps = lut_calib_sweeps
        lut_floor = max(float(lut_floor), 0.0)
        self.lut_floor = lut_floor
        self.lut_default = max(float(lut_default), lut_floor)
        self.lut_decay = lut_decay
        self.lut_warmup_updates = max(int(lut_warmup_updates), 0)
        self.lut_max_flips = lut_max_flips
        self.lut_multiply = float(lut_multiply)
        if lut_max_flips is not None:
            k_max = max(int(lut_max_flips), 0)
        else:
            interval_width = self._compute_interval_width(use_interval_mult=True)
            denom = 2 * interval_width
            if interval_width > 0 and self.hilbert.size % denom == 0:
                interval_count = self.hilbert.size // denom
                k_max = max(interval_count, 0)
            else:
                k_max = max(int(self.sweep_size), 0)

        self.lut_k_max = k_max

        if stagger_pattern is None:
            pattern = jnp.where(
                (jnp.arange(self.hilbert.size) % 2) == 0,
                1.0,
                -1.0,
            )
        else:
            pattern = jnp.asarray(stagger_pattern)
        if pattern.shape[0] != self.hilbert.size:
            raise ValueError(
                f"stagger_pattern must have length {self.hilbert.size}, got {pattern.shape[0]}"
            )
        self._stagger_sign = pattern.astype(jnp.float32)

        if model is not None:
            self.apply_full_headless= nk.utils.HashablePartial(
                model.apply,
                mode="full",
                cache_jac=True,
                mutable=["intermediates", "cache"],
                capture_intermediates=False,
                headless=True,
                remove_shift=True, 
            )

            self.apply_full_headless_no_cache= nk.utils.HashablePartial(
                model.apply,
                mode="full",
                cache_jac=False,
                headless=True,
                remove_shift=True,
            )
        
            self.apply_headless_partial = nk.utils.HashablePartial(
                model.apply,
                mode="partial",
                cache_jac=False,
                headless=True,
                remove_shift=True, 
            )

            self.apply_partial = nk.utils.HashablePartial(
                model.apply,
                mode="partial",
                cache_jac=False,
                headless=False,
                remove_shift=True, 
            )

            self.apply_full = nk.utils.HashablePartial(
                model.apply,
                mode="full",
                cache_jac=False,
                headless=False,
            )

        self.set_sweep_schedule(sweep_ramp)

    def set_sweep_schedule(
        self, sweep_ramp: SweepRampConfig | Callable[[int], int] | None
    ) -> None:
        """
        Configure the sweep-size schedule.
        """
        if sweep_ramp is None:
            self.sweep_schedule = None
            self.sweep_schedule_config = None
            return

        if isinstance(sweep_ramp, SweepRampConfig):
            schedule = sweep_ramp.build(self.base_sweep_size)
            self.sweep_schedule_config = sweep_ramp
        elif callable(sweep_ramp):
            schedule = sweep_ramp
            self.sweep_schedule_config = None
        else:
            raise TypeError(
                "sweep_ramp must be None, a callable, or a SweepRampConfig instance."
            )

        self.sweep_schedule = schedule

    @property
    def has_dynamic_sweep_schedule(self) -> bool:
        return self.sweep_schedule is not None

    def prepare_initial_sweep_state(
        self, state: MetropolisSamplerState
    ) -> MetropolisSamplerState:
        """
        Ensure the sampler state carries the correct sweep target for iteration 0.
        """
        return self.update_typewriter_schedule(state, iteration=0)

    def update_typewriter_schedule(
        self, sampler_state: MetropolisSamplerState, iteration: int
    ) -> MetropolisSamplerState:
        """
        Update the sampler state with the sweep size for the given iteration.
        """
        iteration_val = int(iteration)

        target = self.base_sweep_size
        if self.sweep_schedule is not None:
            target = self.sweep_schedule(iteration_val)

        target = max(int(target), 1)

        current_target = int(np.asarray(sampler_state.typewriter_target_steps))
        current_iteration = int(np.asarray(sampler_state.iteration))

        if current_target == target and current_iteration == iteration_val:
            return sampler_state

        return sampler_state.replace(
            typewriter_target_steps=jnp.asarray(target, dtype=jnp.int32),
            iteration=jnp.asarray(iteration_val, dtype=jnp.int32),
        )

    def make_sweep_schedule_callback(self) -> Callable[[int, Any, Any], None] | None:
        """
        Build a NetKet driver callback that updates the sweep schedule each iteration.
        """
        if self.sweep_schedule is None:
            return None

        def callback(step: int, _log, driver) -> bool:
            vstate = driver.state
            new_state = self.update_typewriter_schedule(
                vstate.sampler_state, iteration=step + 1
            )
            if new_state is not vstate.sampler_state:
                updated_vstate, changed = self._assign_sampler_state(vstate, new_state)
                if changed and updated_vstate is not vstate:
                    driver.state = updated_vstate
            return True

        return callback

    def _assign_sampler_state(self, vstate, new_sampler_state):
        """
        Assign `new_sampler_state` to the NetKet variational state, supporting both
        newer APIs with `.replace` and older ones that require in-place mutation.
        Returns a tuple `(vstate_like, changed)` where `vstate_like` is either the
        updated state (possibly the same object) and `changed` flags whether an
        update occurred.
        """
        if new_sampler_state is vstate.sampler_state:
            return vstate, False

        if hasattr(vstate, "replace"):
            return vstate.replace(sampler_state=new_sampler_state), True

        assigned = False
        try:
            setattr(vstate, "sampler_state", new_sampler_state)
            assigned = True
        except AttributeError:
            try:
                object.__setattr__(vstate, "_sampler_state", new_sampler_state)
                assigned = True
            except AttributeError:
                vstate.__dict__["sampler_state"] = new_sampler_state
                assigned = True

        return vstate, assigned


    @property
    def n_sweeps(self):
        warn_deprecation(
            "`MetropolisSampler.n_sweeps` is deprecated. Please use `MetropolisSampler.sweep_size` instead."
        )
        return self.sweep_size

    def _initialize_lut_fields(self, state: MetropolisSamplerState, *, reset: bool = False) -> MetropolisSamplerState:
        if not self.dyn_tol_enabled:
            return state

        k_max = int(self.lut_k_max)
        values_shape = (k_max + 1,)

        existing_values = state.lut_values
        should_reset = reset or (existing_values is None) or (existing_values.shape != values_shape)

        lut_values = jnp.zeros(values_shape, dtype=jnp.float32)
        lut_counts = jnp.zeros(values_shape, dtype=jnp.int32)
        if should_reset:
            placeholder = jnp.asarray(max(self.lut_floor, 1e-5), dtype=lut_values.dtype)
            lut_values = lut_values.at[0].set(placeholder)
            lut_counts = lut_counts.at[0].set(jnp.asarray(1, dtype=lut_counts.dtype))
        lut_last_update = jnp.asarray(state.n_steps_proc, dtype=jnp.float32)

        new_values = lut_values if should_reset else state.lut_values
        new_counts = lut_counts if should_reset else state.lut_counts

        return state.replace(
            lut_values=new_values,
            lut_counts=new_counts,
            lut_last_update_step=lut_last_update,
            lut_update_count=jnp.asarray(0, dtype=jnp.int32)
            if should_reset
            else state.lut_update_count,
        )

    def _lut_query(
        self,
        state: MetropolisSamplerState,
        k: jnp.ndarray,
    ) -> jnp.ndarray:
        if (not self.dyn_tol_enabled) or state.lut_values is None or state.lut_counts is None:
            fallback = max(float(self.accept_tol), self.lut_floor)
            return jnp.full_like(k, fallback, dtype=jnp.float32)

        lut_values = state.lut_values
        lut_counts = state.lut_counts
        warmup_threshold = jnp.asarray(self.lut_warmup_updates, dtype=state.lut_update_count.dtype)

        k_max = lut_values.shape[0] - 1
        k_idx = jnp.clip(k, 0, k_max).astype(jnp.int32)

        tau = jnp.take(lut_values, k_idx, axis=0)
        counts = jnp.take(lut_counts, k_idx, axis=0)

        default_val = jnp.asarray(self.lut_default, dtype=tau.dtype)
        tau = jnp.where(counts > 0, tau, default_val)

        tau = jnp.where(tau > 0, tau, default_val)
        tau = jnp.maximum(tau, jnp.asarray(self.lut_floor, dtype=tau.dtype))
        tau = jnp.minimum(tau, jnp.asarray(0.25, dtype=tau.dtype))

        default_tau = jnp.full_like(tau, default_val)
        use_default = state.lut_update_count < warmup_threshold
        tau = jnp.where(use_default, default_tau, tau)
        return tau


    def _lut_update(
        self,
        lut_values,
        lut_counts,
        k,
        errors,
        mask=None,
    ):
        if lut_values is None or lut_counts is None:
            return lut_values, lut_counts

        k_max = lut_values.shape[0] - 1
        k_idx = jnp.clip(k, 0, k_max).astype(jnp.int32)

        if mask is None:
            mask = jnp.ones_like(errors, dtype=jnp.bool_)
        else:
            mask = mask.astype(jnp.bool_)

        # only update masked entries; use -inf for others
        err = errors.astype(lut_values.dtype)
        neg_inf = jnp.array(-jnp.inf, dtype=lut_values.dtype)
        upd = jnp.where(mask, err, neg_inf)

        lut_values = lut_values.at[k_idx].max(upd)

        # counts can still just add (duplicates accumulate naturally)
        lut_counts = lut_counts.at[k_idx].add(mask.astype(lut_counts.dtype))

        # enforce monotonic increase w.r.t accepted flips along axis 0
        lut_values = self._ensure_lut_monotonic(lut_values)

        return lut_values, lut_counts


    def _ensure_lut_monotonic(self, lut_values: jnp.ndarray | None) -> jnp.ndarray | None:
        if lut_values is None:
            return None
        if lut_values.shape[0] <= 1:
            return lut_values

        first_plane = lut_values[0]

        def monotonic_scan(carry, current):
            updated = jnp.maximum(carry, current)
            return updated, updated

        _, tail = jax.lax.scan(monotonic_scan, first_plane, lut_values[1:])

        return jnp.concatenate((first_plane[jnp.newaxis, ...], tail), axis=0)


    def _apply_lut_decay(self, lut_values: jnp.ndarray | None) -> jnp.ndarray | None:
        if lut_values is None:
            return None
        decay = jnp.asarray(self.lut_decay, dtype=lut_values.dtype)
        return lut_values * decay

    def _compute_interval_width(self, use_interval_mult: bool = True) -> int:
        """
        Compute the interval width for typewriter sampling, inflated by the
        radius of the local move rule.
        """
        base = self.token_size * (2 * (self.width // 2) + 1)
        if use_interval_mult:
            base = self.interval_mult * base
        radius = int(getattr(self.local_move_rule, "radius", 0))
        target = int(base + radius)

        L = int(self.hilbert.size)
        half_L = L // 2

        if target <= 0:
            raise ValueError(
                f"Requested interval width {target} invalid for L={L}; "
                f"check token_size {self.token_size}, width {self.width}, interval_mult {self.interval_mult}, "
                f"and move radius {radius}."
            )

        # Choose the smallest divisor of half_L that is >= target.
        # This is half_L // floor(half_L / target); if target exceeds half_L,
        # the floor is zero so we clamp the denominator to 1 (width=half_L).
        k = max(half_L // target, 1)
        width = half_L // k

        # Guarantee divisibility of the full chain length.
        width = int(width)
        if L % (2 * width) != 0:
            # Ceil up to the next valid divisor analytically (no loops).
            k = max((L + 2 * width - 1) // (2 * width), 1)
            width = L // (2 * k)

        return width

    def _random_state_in_sector(self, key, batch_size: int):
        """
        Generate random ±1 configurations with fixed total magnetisation.
        """
        L = int(self.hilbert.size)
        m = int(self.sz_sector)

        if abs(m) > L or (L + m) % 2 != 0:
            raise ValueError(
                f"Requested sz_sector={m} incompatible with system size {L}"
            )

        n_up = (L + m) // 2  # number of +1 spins
        if n_up == 0:
            return -jnp.ones((batch_size, L), dtype=self.dtype)
        if n_up == L:
            return jnp.ones((batch_size, L), dtype=self.dtype)

        # Use a single RNG draw and top_k to select the +1 positions per batch.
        scores = jax.random.uniform(key, (batch_size, L))
        _, top_idx = jax.lax.top_k(scores, n_up)
        spins = jnp.full((batch_size, L), -1, dtype=self.dtype)
        row_idx = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
        spins = spins.at[row_idx, top_idx].set(jnp.asarray(1, dtype=spins.dtype))
        return spins

    def sample_next(
        self,
        machine: Callable | nn.Module,
        parameters: PyTree,
        state: SamplerState | None = None,
    ) -> tuple[SamplerState, jnp.ndarray]:
        """
        Samples the next state in the Markov chain.

        Args:
            machine: A Flax module or callable with the forward pass of the log-pdf.
                If it is a callable, it should have the signature
                :code:`f(parameters, σ) -> jnp.ndarray`.
            parameters: The PyTree of parameters of the model.
            state: The current state of the sampler. If not specified, then initialize and reset it.

        Returns:
            state: The new state of the sampler.
            σ: The next batch of samples.

        Note:
            The return order is inverted wrt `sample` because when called inside of
            a scan function the first returned argument should be the state.
        """
        if state is None:
            state = self.reset(machine, parameters)

        sample_fun = self._sample_next_super_fast_auto_stop if self.sample_fast else self._sample_next
        return sample_fun(wrap_afun(machine), parameters, state)

    @partial(jax.jit, static_argnums=1)
    def _init_state(self, machine, parameters, key):
        key_state, key_rule = jax.random.split(key)
        rule_state = self.rule.init_state(self, machine, parameters, key_rule)
        σ = jnp.zeros((self.n_batches, self.hilbert.size), dtype=self.dtype)
        σ = shard_along_axis(σ, axis=0)

        output_dtype = jax.eval_shape(machine.apply, parameters, σ).dtype
        log_prob = jnp.full((self.n_batches,), -jnp.inf, dtype=dtype_real(output_dtype))
        log_prob = shard_along_axis(log_prob, axis=0)

        state = MetropolisSamplerState(
            σ=σ, rng=key_state, rule_state=rule_state, log_prob=log_prob
        )
        state = state.replace(
            typewriter_target_steps=jnp.asarray(self.base_sweep_size, dtype=jnp.int32),
            iteration=jnp.asarray(0, dtype=jnp.int32),
        )
        # If we don't reset the chain at every sampling iteration, then reset it
        # now.
        if self.dyn_tol_enabled:
            state = self._initialize_lut_fields(state, reset=True)

        if not self.reset_chains:
            key_state, rng = jax.jit(jax.random.split)(key_state)
            if self.sz_sector is None:
                σ = self.rule.random_state(self, machine, parameters, state, rng)
            else:
                σ = self._random_state_in_sector(rng, self.n_batches)
            _assert_good_sample_shape(
                σ,
                (self.n_batches, self.hilbert.size),
                self.dtype,
                f"{self.rule}.random_state",
            )
            σ = shard_along_axis(σ, axis=0)
            state = state.replace(σ=σ, rng=key_state)
        return state

    @partial(jax.jit, static_argnums=1)
    def _reset(self, machine, parameters, state):
        rng = state.rng

        if self.reset_chains:
            rng, key = jax.random.split(state.rng)
            if self.sz_sector is None:
                σ = self.rule.random_state(self, machine, parameters, state, key)
            else:
                σ = self._random_state_in_sector(key, self.n_batches)
            _assert_good_sample_shape(
                σ,
                (self.n_batches, self.hilbert.size),
                self.dtype,
                f"{self.rule}.random_state",
            )
            σ = shard_along_axis(σ, axis=0)
        else:
            σ = state.σ

        # Recompute the log_probability of the current samples
        if self.sample_fast: 
            # If we are using the fast sampling method, we need to compute the log probability
            # of the current samples using the full machine evaluation.
            apply_machine = self.apply_full_headless_no_cache
            vector = apply_machine(parameters, σ)
            log_prob_σ = self.machine_pow * logcosh_activation(vector).real
            _assert_good_log_prob_shape(log_prob_σ, self.n_batches, machine)
            
            rule_state = self.rule.reset(self, machine, parameters, state)
            state = state.replace(
                σ=σ,
                log_prob=log_prob_σ,
                rng=rng,
                rule_state=rule_state,
                n_steps_proc=jnp.zeros_like(state.n_steps_proc),
                n_accepted_proc=jnp.zeros_like(state.n_accepted_proc),
                interval_index=jnp.zeros_like(log_prob_σ, dtype=jnp.int32),
                freeze_correction=jnp.ones_like(state.freeze_correction),
            )
            if self.dyn_tol_enabled:
                state = self._initialize_lut_fields(state, reset=True)
            return state

        else: 
        
            apply_machine = apply_chunked(
                machine.apply, in_axes=(None, 0), chunk_size=self.chunk_size
            )
            log_prob_σ = self.machine_pow * apply_machine(parameters, σ).real

            rule_state = self.rule.reset(self, machine, parameters, state)
            state = state.replace(
                σ=σ,
                log_prob=log_prob_σ,
                rng=rng,
                rule_state=rule_state,
                n_steps_proc=jnp.zeros_like(state.n_steps_proc),
                n_accepted_proc=jnp.zeros_like(state.n_accepted_proc),
                freeze_correction=jnp.ones_like(state.freeze_correction),
            )
            if self.dyn_tol_enabled:
                state = self._initialize_lut_fields(state, reset=True)
            return state

    def _sample_next(self, machine, parameters, state):
        """
        Implementation of `sample_next` for subclasses of `MetropolisSampler`.

        If you subclass `MetropolisSampler`, you should override this and not `sample_next`
        itself, because `sample_next` contains some common logic.
        """
        apply_machine = apply_chunked(
            machine.apply, in_axes=(None, 0), chunk_size=self.chunk_size
        )

        def loop_body(i, s):
            # 1 to propagate for next iteration, 1 for uniform rng and n_chains for transition kernel
            s["key"], key1, key2 = jax.random.split(s["key"], 3)

            σp, log_prob_correction = self.rule.transition(
                self, machine, parameters, state, key1, s["σ"]
            )
            _assert_good_sample_shape(
                σp,
                (self.n_batches, self.hilbert.size),
                self.dtype,
                f"{self.rule}.transition",
            )
            proposal_log_prob = self.machine_pow * apply_machine(parameters, σp).real
            _assert_good_log_prob_shape(proposal_log_prob, self.n_batches, machine)

            uniform = jax.random.uniform(key2, shape=(self.n_batches,))
            if log_prob_correction is not None:
                do_accept = uniform < jnp.exp(
                    proposal_log_prob - s["log_prob"] + log_prob_correction
                )
            else:
                do_accept = uniform < jnp.exp(proposal_log_prob - s["log_prob"])

            # do_accept must match ndim of proposal and state (which is 2)
            s["σ"] = jnp.where(do_accept.reshape(-1, 1), σp, s["σ"])
            s["accepted"] += do_accept

            s["log_prob"] = jnp.where(
                do_accept.reshape(-1), proposal_log_prob, s["log_prob"]
            )

            return s

        s = {
            "key": state.rng,
            "σ": state.σ,
            # Log prob is already computed in reset, so don't recompute it.
            # "log_prob": self.machine_pow * apply_machine(parameters, state.σ).real,
            "log_prob": state.log_prob,
            # for logging
            "accepted": state.n_accepted_proc,
        }
        s = jax.lax.fori_loop(0, self.sweep_size, loop_body, s)

        new_state = state.replace(
            rng=s["key"],
            σ=s["σ"],
            log_prob=s["log_prob"],
            n_accepted_proc=s["accepted"],
            n_steps_proc=state.n_steps_proc + self.sweep_size * self.n_batches,
        )

        return new_state, (new_state.σ, new_state.log_prob)
    
    

    @partial(jax.jit, static_argnums=(1, 4))
    def _sample_typewriter(self, machine, parameters, state, debug):
        """
        Implementation of `sample_next` for subclasses of `MetropolisSampler`.

        If you subclass `MetropolisSampler`, you should override this and not `sample_next`
        itself, because `sample_next` contains some common logic.
        """
        assert self.chunk_size is None, "Chunk size not implemented yet"

        # full eval → (vector, cache)
        apply_machine_full = self.apply_full_headless

        # unpack incoming state
        key       = state.rng
        σ_current = state.σ
        logp_current = state.log_prob
        accepted = state.n_accepted_proc
        interval_index = state.interval_index
        interval_width = self._compute_interval_width(use_interval_mult=True)

        assert σ_current.shape[-1] % (2* interval_width) == 0, f"Hilbert space size {σ_current.shape[-1]} must be divisible by interval width {interval_width}"
        interval_count = σ_current.shape[-1] // interval_width // 2
        batch_size = σ_current.shape[0]

        # full machine evaluation
        vector, cache = apply_machine_full(parameters, σ_current)
        #cache = jax.tree.map(_cast_to_bf16, cache)

        # extract log‐prob
        logp_current = self.machine_pow * logcosh_activation(vector).real

        paramDict = {"params": parameters["params"], **cache}

        # Generate the center arrays
        key, key_centers, key_moves_base, key_uniforms = jax.random.split(key, 4)
        relative_centers = jax.random.randint(key_centers, (interval_count, batch_size,), 0, interval_width)
        center_offset_index = interval_index[None, :] + jnp.arange(0, 2 * interval_count, 2)[:, None] 
        centers = relative_centers + interval_width * center_offset_index
        key_moves = jax.random.split(key_moves_base, interval_count)
        uniforms = jax.random.uniform(key_uniforms, shape=(interval_count, batch_size))
        indices = jnp.arange(interval_count, dtype=jnp.int8)
        idx = jnp.arange(batch_size, dtype=jnp.int32)

        # This function slices the array 
        def _one_sweep(σ, centers, key_move):
            """
            All computations that depended only on (key1, key2, σ_current)
            are pure and can therefore be vmapped.
            """
            σp = self.local_move_rule.apply(σ, centers, key_move)
            σp_sliced = slice_tokens_jit(
                σp, idx, centers, width=self.width, token_size=self.token_size
            )
            return σp_sliced

        # Broadcast σ_current once; map over the key axes (axis-0)
        σ_prop_chain_sliced  = jax.vmap(_one_sweep, in_axes=(None, 0, 0))(
            σ_current, centers, key_moves
        )

        # Apply the actual network 
        apply_func = jax.vmap(
            lambda p, s, id, c: self.apply_headless_partial(p, s, centers=(id, c)), in_axes=(None, 0, None, 0), out_axes=0
        )
        vectorsNewFull = apply_func(paramDict, σ_prop_chain_sliced, idx, centers)
        vectorsNew = vectorsNewFull - vector[None, :]  # Subtract the base vector
        
        # Prepare the carry for history replay
        # carry definition
        class ReplayCarry(NamedTuple):
            σ: jnp.ndarray
            σ_prop: jnp.ndarray
            log_prob: jnp.ndarray
            freeze_iter: jnp.ndarray
            accepted: jnp.ndarray
            current_vector: jnp.ndarray
            interval_index: jnp.ndarray
            stored_u: jnp.ndarray
            stored_a: jnp.ndarray
            stored_mask: jnp.ndarray
            lut_values: Any
            lut_counts: Any
            last_update: jnp.ndarray
            lut_update_count: jnp.ndarray

        class ReplayCarryDebug(NamedTuple):
            σ: jnp.ndarray
            σ_prop: jnp.ndarray
            log_prob: jnp.ndarray
            freeze_iter: jnp.ndarray
            accepted: jnp.ndarray
            current_vector: jnp.ndarray
            interval_index: jnp.ndarray
            stored_u: jnp.ndarray
            stored_a: jnp.ndarray
            stored_mask: jnp.ndarray
            lut_values: Any
            lut_counts: Any
            last_update: jnp.ndarray
            lut_update_count: jnp.ndarray
            debug_errors: jnp.ndarray
            debug_wrong: jnp.ndarray
            debug_total_error: jnp.ndarray
            tau_dbg: jnp.ndarray

        carry_cls = ReplayCarryDebug if debug else ReplayCarry
        init_carry = carry_cls(
            σ=σ_current,
            σ_prop=σ_current,
            log_prob=logp_current,
            freeze_iter=jnp.full_like(logp_current, interval_count, dtype=jnp.int32),
            accepted=jnp.zeros_like(accepted, dtype=jnp.int32),
            current_vector=vector,
            interval_index=interval_index.astype(jnp.int32),
            stored_u=jnp.ones_like(logp_current, dtype =jnp.float32),
            stored_a=jnp.zeros_like(logp_current, dtype=jnp.float32),
            stored_mask=jnp.zeros_like(logp_current, dtype=jnp.bool_),
            lut_values=state.lut_values,
            lut_counts=state.lut_counts,
            last_update=state.lut_last_update_step,
            lut_update_count=state.lut_update_count,
            **(
                {}
                if not debug
                else dict(
                    debug_errors=jnp.zeros((interval_count, batch_size), dtype=jnp.float32),
                    debug_wrong=jnp.zeros_like(logp_current, dtype=jnp.int32),
                    debug_total_error=jnp.zeros_like(logp_current, dtype=jnp.float32),
                    tau_dbg=jnp.zeros((interval_count, batch_size), dtype=jnp.float32),
                )
            ),
        )

        scan_inputs_full = (vectorsNew, centers, key_moves, uniforms, indices)

        # scan step
        def scan_step(s, inputs):
            vectorNew_i, centers_i, key_move_i, uniform_i, step_idx = inputs

            σp = self.local_move_rule.apply(s.σ, centers_i, key_move_i)

            proposal_log_prob = self.machine_pow * logcosh_activation(s.current_vector + vectorNew_i).real
            metropolis_frac = jnp.minimum(jnp.exp(proposal_log_prob - s.log_prob), 1.0)

            dyn_active = self.dyn_tol_enabled and state.lut_values is not None
            if dyn_active:
                tau = self.lut_multiply * self._lut_query(state, s.accepted).astype(metropolis_frac.dtype)
            else:
                tau = self.accept_tol

            is_frozen = s.freeze_iter < interval_count
            is_close = jnp.abs(uniform_i - metropolis_frac) < tau
            accept_stop = s.accepted >= self.max_accept
            is_close = jnp.logical_or(is_close, accept_stop) & ~is_frozen
            in_chain = s.interval_index < 2 * interval_count
            is_close = jnp.logical_and(is_close, in_chain)

            # Do not freeze if nothing has been accepted yet
            is_close = jnp.logical_and(is_close, s.accepted > 0)

            σ_prop = jnp.where(is_close[:, None], σp, s.σ_prop)
            freeze_iter = jnp.where(is_close, step_idx, s.freeze_iter)
            stored_u = jnp.where(is_close, uniform_i, s.stored_u)
            stored_a = jnp.where(is_close, metropolis_frac, s.stored_a)
            stored_mask = jnp.where(
                is_close, jnp.ones_like(is_close, dtype=jnp.bool_), s.stored_mask
            )

            if (not dyn_active) and (not debug):
                stored_a = s.stored_a
                stored_mask = s.stored_mask

            if debug:
                exact_log_prob = self.machine_pow * self.apply_full(parameters, σp).real
                exact_prev_log_prob = self.machine_pow * self.apply_full(parameters, s.σ).real

                metropolis_exact = jnp.minimum(jnp.exp(exact_log_prob - exact_prev_log_prob), 1.0)
                approx_accept = uniform_i < metropolis_frac
                exact_accept = uniform_i < metropolis_exact
                wrong = approx_accept != exact_accept
                error_raw = jnp.abs(metropolis_exact - metropolis_frac)
                debug_errors = s.debug_errors.at[step_idx].set(error_raw.astype(jnp.float32))
                delta_error = jnp.where(is_close, 0.0, error_raw)
                debug_total_error = s.debug_total_error + delta_error.astype(jnp.float32)
                debug_wrong = s.debug_wrong + jnp.where(
                    jnp.logical_or(jnp.logical_or(is_close, is_frozen), ~in_chain),
                    0,
                    wrong.astype(s.debug_wrong.dtype),
                )
                tau_dbg = s.tau_dbg.at[step_idx].set(tau)
            else:
                debug_errors = None
                debug_total_error = None
                debug_wrong = None
                tau_dbg = None

            do_accept = (uniform_i < metropolis_frac) & ~is_close & ~is_frozen
            do_accept = jnp.logical_and(do_accept, in_chain)

            current_vector = jnp.where(
                do_accept.reshape(-1, 1), s.current_vector + vectorNew_i, s.current_vector
            )
            σ_new = jnp.where(do_accept.reshape(-1, 1), σp, s.σ)
            accepted_new = s.accepted + do_accept

            unfrozen = ~is_frozen & ~is_close & in_chain
            interval_index = s.interval_index + 2 * unfrozen

            log_prob = jnp.where(do_accept.reshape(-1), proposal_log_prob, s.log_prob)

            new_s = carry_cls(
                σ=σ_new,
                σ_prop=σ_prop,
                log_prob=log_prob,
                freeze_iter=freeze_iter,
                accepted=accepted_new,
                current_vector=current_vector,
                interval_index=interval_index,
                stored_u=stored_u,
                stored_a=stored_a,
                stored_mask=stored_mask,
                lut_values=s.lut_values,
                lut_counts=s.lut_counts,
                last_update=s.last_update,
                lut_update_count=s.lut_update_count,
                **(
                    {}
                    if not debug
                    else dict(
                        debug_errors=debug_errors,
                        debug_wrong=debug_wrong,
                        debug_total_error=debug_total_error,
                        tau_dbg=tau_dbg,
                    )
                ),
            )

            return new_s, do_accept

        s, proposal_accepted = jax.lax.scan(
            scan_step, init_carry, scan_inputs_full, unroll=4
        )
        proposal_accepted = proposal_accepted.astype(jnp.int32)
        key, key2 = jax.random.split(key, 2)
        
        # Propose the updates, propose idenical state for unfrozen chains
        is_frozen = s.freeze_iter < interval_count
        proposal = jnp.where(
            is_frozen[:, None], s.σ_prop,  s.σ
        )

        # Calculate log prob exactly via more expensive full evaluation
        proposal_log_prob = self.machine_pow * self.apply_full(parameters, proposal).real

        # Compare to the saved uniform value for metropolis acceptance
        uniform = s.stored_u
        exact_frac = jnp.clip(jnp.exp(proposal_log_prob - s.log_prob), 0.0, 1.0)
        do_accept = uniform < exact_frac
        do_accept = jnp.logical_and(do_accept, is_frozen)

        # Update only the ones that are inside the chain 
        in_chain = s.interval_index < 2 * interval_count
        do_accept = jnp.logical_and(do_accept, in_chain)

        accepted_final = s.accepted + do_accept

        freeze_sum = jnp.sum(s.freeze_iter)
        delta_steps_val = freeze_sum
        mean_freeze = jnp.mean(s.freeze_iter)
        new_correction = interval_count / jnp.maximum(mean_freeze, 1e-2)
        new_correction = jnp.maximum(new_correction, 1.0)

        if self.dyn_tol_enabled and s.lut_values is not None:
            approx_frac = s.stored_a
            mask = s.stored_mask & is_frozen
            error = jnp.abs(exact_frac.astype(jnp.float32) - approx_frac)
            threshold = jnp.asarray(
                self.lut_refresh_every * self.n_batches * self.lut_calib_sweeps,
                dtype=jnp.int32,
            )
            delta_steps_i32 = delta_steps_val.astype(jnp.int32)
            steps_total = jnp.asarray(state.n_steps_proc, dtype=jnp.int32) + delta_steps_i32
            steps_since = steps_total - s.last_update
            needs_decay = steps_since >= threshold

            lut_values = jax.lax.cond(
                needs_decay,
                lambda v: self._apply_lut_decay(v),
                lambda v: v,
                s.lut_values,
            )
            lut_values, lut_counts = self._lut_update(
                lut_values,
                s.lut_counts,
                accepted_final,
                error,
                mask=mask,
            )
            lut_last_update = jnp.where(needs_decay, steps_total, s.last_update)
            lut_update_count = s.lut_update_count + jnp.asarray(
                jnp.any(mask), dtype=s.lut_update_count.dtype
            )
        else:
            lut_values = s.lut_values
            lut_counts = s.lut_counts
            lut_last_update = s.last_update
            lut_update_count = s.lut_update_count

        # Update the accepted vectors
        σ_final = jnp.where(do_accept.reshape(-1, 1), proposal, s.σ)
        interval_index_final = s.interval_index + 2 * is_frozen

        log_prob_final = jnp.where(
            do_accept.reshape(-1), proposal_log_prob, s.log_prob
        )

        accepted = accepted + accepted_final
        delta_steps = delta_steps_val

        finished_chain = ~(interval_index_final < 2 * interval_count)

        update_interval_index = (interval_index_final + 1) % 2
        interval_index = jnp.where(finished_chain, update_interval_index, interval_index_final)

        new_state = state.replace(
            rng=key,
            σ=σ_final,
            log_prob=log_prob_final,
            n_accepted_proc= accepted,
            interval_index=interval_index,
            n_steps_proc=state.n_steps_proc + delta_steps,  # + (1 + self.sweep_size) * self.n_batches
            lut_values=lut_values if self.dyn_tol_enabled else state.lut_values,
            lut_counts=lut_counts if self.dyn_tol_enabled else state.lut_counts,
            lut_last_update_step=lut_last_update if self.dyn_tol_enabled else state.lut_last_update_step,
            lut_update_count=lut_update_count if self.dyn_tol_enabled else state.lut_update_count,
            freeze_correction=new_correction,
        )

        debug_payload = None
        if debug:
            proposal_accepted_debug = proposal_accepted
            prop_accepted_update = jnp.where(
                s.freeze_iter < interval_count,
                do_accept,
                proposal_accepted_debug[s.freeze_iter, idx],
            )
            proposal_accepted_debug = proposal_accepted_debug.at[s.freeze_iter, idx].set(prop_accepted_update)

            # debug-only attachments
            debug_payload = {
                "σ": σ_final,
                "σ_prop": s.σ_prop,
                "proposal_centers": centers,
                "log_prob": log_prob_final,
                "freeze_iter": s.freeze_iter,
                "accepted": accepted_final,
                "replay_vectors": vectorsNew,
                "replay_vectors_debug": vectorsNewFull,
                "current_vector": s.current_vector,
                "proposal_accepted": proposal_accepted_debug,
                "interval_index": interval_index_final,
                "stored_u": s.stored_u,
                "stored_a": s.stored_a,
                "stored_mask": s.stored_mask,
                "lut_values": lut_values,
                "lut_counts": lut_counts,
                "last_update": lut_last_update,
                "lut_update_count": lut_update_count,
                "debug_errors": s.debug_errors,
                "debug_wrong": s.debug_wrong,
                "debug_total_error": s.debug_total_error,
                "tau_dbg": s.tau_dbg,
            }

        if self.global_proposal: 
            new_state =  self._propose_global_update(
                machine, parameters, new_state, key2, can_accept=finished_chain
            )

        if debug:
            return new_state, (new_state.σ, new_state.log_prob, ), debug_payload

        return new_state, (new_state.σ, new_state.log_prob, )

    @partial(jax.jit, static_argnums=(1, 4))
    def _sample_typewriter_shift_project(self, machine, parameters, state, debug):
        """
        Typewriter sampler that keeps cached tensors for both the base input and
        the configuration rolled by one site when shift projection is enabled.
        """
        assert self.chunk_size is None, "Chunk size not implemented yet"

        apply_machine_full = self.apply_full_headless

        key = state.rng
        σ_current = state.σ
        accepted = state.n_accepted_proc
        interval_index = state.interval_index

        interval_width = self._compute_interval_width(use_interval_mult=True)
        assert (
            σ_current.shape[-1] % (2 * interval_width) == 0
        ), f"Hilbert space size {σ_current.shape[-1]} must be divisible by interval width {interval_width}"
        interval_count = σ_current.shape[-1] // interval_width // 2
        batch_size = σ_current.shape[0]

        vector, cache = apply_machine_full(parameters, σ_current)
        σ_shift = jnp.roll(σ_current, 1, axis=-1)
        vector_shift, cache_shift = apply_machine_full(parameters, σ_shift)

        logcosh_main = logcosh_activation(vector)
        logcosh_shift = logcosh_activation(vector_shift)
        logp_current = self.machine_pow * (
            jnp.logaddexp(logcosh_main, logcosh_shift) - jnp.log(2.0)
        ).real

        param_dict = {"params": parameters["params"], **cache}
        param_dict_shift = {"params": parameters["params"], **cache_shift}

        key, key_centers, key_moves_base = jax.random.split(key, 3)
        relative_centers = jax.random.randint(
            key_centers, (interval_count, batch_size,), 0, interval_width
        )
        center_offset_index = interval_index[None, :] + jnp.arange(0, 2 * interval_count, 2)[:, None]
        centers = relative_centers + interval_width * center_offset_index
        centers_shift = (centers + 1) % σ_current.shape[-1]
        key_moves = jax.random.split(key_moves_base, interval_count)
        idx = jnp.arange(batch_size, dtype=jnp.int32)

        def _one_sweep(σ, centers_local, key_move):
            σp_local = self.local_move_rule.apply(σ, centers_local, key_move)
            σp_sliced = slice_tokens_jit(
                σp_local, idx, centers_local, width=self.width, token_size=self.token_size
            )
            return σp_sliced

        σ_prop_chain_sliced = jax.vmap(_one_sweep, in_axes=(None, 0, 0))(
            σ_current, centers, key_moves
        )
        σ_prop_chain_sliced_shift = jax.vmap(_one_sweep, in_axes=(None, 0, 0))(
            σ_shift, centers_shift, key_moves
        )

        apply_func = jax.vmap(
            lambda p, s, id, c: self.apply_headless_partial(p, s, centers=(id, c)),
            in_axes=(None, 0, None, 0),
            out_axes=0,
        )

        vectors_new_full = apply_func(param_dict, σ_prop_chain_sliced, idx, centers)
        vectors_new_full_shift = apply_func(
            param_dict_shift, σ_prop_chain_sliced_shift, idx, centers_shift
        )
        vectors_new = vectors_new_full - vector[None, :]  # Subtract the base vector
        vectors_new_shift = vectors_new_full_shift - vector_shift[None, :]

        carry_replay = {
            "key": key,
            "σ": σ_current,
            "σ_shift": σ_shift,
            "σ_prop": σ_current,
            "σ_prop_shift": σ_shift,
            "proposal_centers": centers,
            "proposal_centers_shift": centers_shift,
            "log_prob": logp_current,
            "freeze_iter": jnp.full_like(logp_current, interval_count, dtype=jnp.int32),
            "accepted": jnp.zeros_like(accepted, dtype=jnp.int32),
            "replay_vectors": vectors_new,
            "replay_vectors_shift": vectors_new_shift,
            "replay_vectors_debug": vectors_new_full,
            "replay_vectors_shift_debug": vectors_new_full_shift,
            "current_vector": vector,
            "current_vector_shift": vector_shift,
            "proposal_accepted": jnp.zeros_like(centers, dtype=jnp.int32),
            "interval_index": interval_index,
            "stored_u": jnp.ones_like(logp_current),
            "stored_a": jnp.zeros_like(logp_current, dtype=jnp.float32),
            "stored_mask": jnp.zeros_like(logp_current, dtype=jnp.bool_),
            "lut_values": state.lut_values,
            "lut_counts": state.lut_counts,
            "last_update": state.lut_last_update_step,
            "lut_update_count": state.lut_update_count,
            "proposal_keys": key_moves,
        }
        if debug:
            carry_replay["debug_errors"] = jnp.zeros((interval_count, batch_size), dtype=jnp.float32)
            carry_replay["debug_wrong"] = jnp.zeros_like(logp_current, dtype=jnp.int32)
            carry_replay["debug_total_error"] = jnp.zeros_like(logp_current, dtype=jnp.float32)
            carry_replay["tau_dbg"] = jnp.zeros((interval_count, batch_size), dtype=jnp.float32)

        def loop_history_replay(i, s):
            key2, next_key = jax.random.split(s["key"], 2)
            s["key"] = next_key
            uniform = jax.random.uniform(key2, shape=(batch_size,))

            interval_index_local = s["interval_index"]
            centers_local = s["proposal_centers"][i]
            centers_shift_local = s["proposal_centers_shift"][i]
            key_move = s["proposal_keys"][i]

            σp = self.local_move_rule.apply(s["σ"], centers_local, key_move)
            σp_shift = self.local_move_rule.apply(s["σ_shift"], centers_shift_local, key_move)

            vector_new = s["replay_vectors"][i]
            vector_new_shift = s["replay_vectors_shift"][i]
            log_prob_new_main = logcosh_activation(s["current_vector"] + vector_new)
            log_prob_new_shift = logcosh_activation(s["current_vector_shift"] + vector_new_shift)
            proposal_log_prob = self.machine_pow * (
                jnp.logaddexp(log_prob_new_main, log_prob_new_shift) - jnp.log(2.0)
            ).real

            metropolis_frac = jnp.minimum(jnp.exp(proposal_log_prob - s["log_prob"]), 1.0)

            dyn_active = self.dyn_tol_enabled and state.lut_values is not None
            if dyn_active:
                tau = self.lut_multiply * self._lut_query(state, s["accepted"]).astype(metropolis_frac.dtype)
            else:
                tau = self.accept_tol

            is_frozen = s["freeze_iter"] < interval_count
            is_close = (jnp.abs(uniform - metropolis_frac) < tau)
            accept_stop = s["accepted"] >= self.max_accept
            is_close = jnp.logical_or(is_close, accept_stop) & ~is_frozen
            in_chain = interval_index_local < 2 * interval_count
            is_close = jnp.logical_and(is_close, in_chain)
            is_close = jnp.logical_and(is_close, s["accepted"] > 0)

            s["σ_prop"] = jnp.where(is_close[:, None], σp, s["σ_prop"])
            s["σ_prop_shift"] = jnp.where(is_close[:, None], σp_shift, s["σ_prop_shift"])
            s["freeze_iter"] = jnp.where(is_close, i, s["freeze_iter"])
            s["stored_u"] = jnp.where(is_close, uniform, s["stored_u"])
            if dyn_active or debug:
                stored_a = metropolis_frac.astype(jnp.float32)
                s["stored_a"] = jnp.where(is_close, stored_a, s["stored_a"])
                s["stored_mask"] = jnp.where(is_close, jnp.ones_like(is_close, dtype=jnp.bool_), s["stored_mask"])

            if debug:
                exact_log_prob = self.machine_pow * self.apply_full(parameters, σp).real
                exact_prev_log_prob = self.machine_pow * self.apply_full(parameters, s["σ"]).real

                metropolis_exact = jnp.minimum(jnp.exp(exact_log_prob - exact_prev_log_prob), 1.0)
                approx_accept = uniform < metropolis_frac
                exact_accept = uniform < metropolis_exact
                wrong = approx_accept != exact_accept
                error_raw = jnp.abs(metropolis_exact - metropolis_frac)
                s["debug_errors"] = s["debug_errors"].at[i].set(error_raw.astype(jnp.float32))
                delta_error = jnp.where(is_close, 0.0, error_raw)
                s["debug_total_error"] = s["debug_total_error"] + delta_error.astype(jnp.float32)
                s["debug_wrong"] = s["debug_wrong"] + jnp.where(
                    jnp.logical_or(jnp.logical_or(is_close, is_frozen), ~in_chain), 0, wrong.astype(s["debug_wrong"].dtype)
                )

            do_accept = (uniform < metropolis_frac) & ~is_close & ~is_frozen
            do_accept = jnp.logical_and(do_accept, in_chain)

            s["current_vector"] = jnp.where(do_accept.reshape(-1, 1), s["current_vector"] + vector_new, s["current_vector"])
            s["current_vector_shift"] = jnp.where(
                do_accept.reshape(-1, 1), s["current_vector_shift"] + vector_new_shift, s["current_vector_shift"]
            )
            s["σ"] = jnp.where(do_accept.reshape(-1, 1), σp, s["σ"])
            s["σ_shift"] = jnp.where(do_accept.reshape(-1, 1), σp_shift, s["σ_shift"])
            s["accepted"] += do_accept
            s["proposal_accepted"] = s["proposal_accepted"].at[i].set(do_accept)

            unfrozen = ~is_frozen & ~is_close & in_chain
            s["interval_index"] += 2 * unfrozen

            s["log_prob"] = jnp.where(
                do_accept.reshape(-1), proposal_log_prob, s["log_prob"]
            )

            if debug:
                s["tau_dbg"] = s["tau_dbg"].at[i].set(tau)

            return s

        s = jax.lax.fori_loop(0, interval_count, loop_history_replay, carry_replay)
        key, key2 = jax.random.split(s["key"], 2)

        is_frozen = s["freeze_iter"] < interval_count
        proposal = jnp.where(
            is_frozen[:, None], s["σ_prop"], s["σ"]
        )
        proposal_shift = jnp.where(
            is_frozen[:, None], s["σ_prop_shift"], s["σ_shift"]
        )

        proposal_log_prob = self.machine_pow * self.apply_full(parameters, proposal).real 

        uniform = s["stored_u"]
        exact_frac = jnp.clip(jnp.exp(proposal_log_prob - s["log_prob"]), 0.0, 1.0)
        do_accept = uniform < exact_frac
        do_accept = jnp.logical_and(do_accept, is_frozen)

        in_chain = s["interval_index"] < 2 * interval_count
        do_accept = jnp.logical_and(do_accept, in_chain)

        freeze_sum = jnp.sum(s["freeze_iter"])
        delta_steps_val = freeze_sum
        mean_freeze = np.mean(s["freeze_iter"])
        new_correction = interval_count / jnp.maximum(mean_freeze, 1e-2)
        new_correction = jnp.maximum(new_correction, 1.0)

        if self.dyn_tol_enabled and s["lut_values"] is not None:
            approx_frac = s["stored_a"]
            mask = s["stored_mask"] & is_frozen
            error = jnp.abs(exact_frac.astype(jnp.float32) - approx_frac)
            threshold = jnp.asarray(
                self.lut_refresh_every * self.n_batches * self.lut_calib_sweeps,
                dtype=jnp.int32,
            )
            delta_steps_i32 = delta_steps_val.astype(jnp.int32)
            steps_total = jnp.asarray(state.n_steps_proc, dtype=jnp.int32) + delta_steps_i32
            steps_since = steps_total - s["last_update"]
            needs_decay = steps_since >= threshold

            lut_values = jax.lax.cond(
                needs_decay,
                lambda v: self._apply_lut_decay(v),
                lambda v: v,
                s["lut_values"],
            )
            lut_values, lut_counts = self._lut_update(
                lut_values,
                s["lut_counts"],
                s["accepted"],
                error,
                mask=mask,
            )
            s["lut_values"] = lut_values
            s["lut_counts"] = lut_counts
            s["last_update"] = jnp.where(needs_decay, steps_total, s["last_update"])
            update_delta = jnp.asarray(jnp.any(mask), dtype=s["lut_update_count"].dtype)
            s["lut_update_count"] = s["lut_update_count"] + update_delta

        s["σ"] = jnp.where(do_accept.reshape(-1, 1), proposal, s["σ"])
        s["σ_shift"] = jnp.where(do_accept.reshape(-1, 1), proposal_shift, s["σ_shift"])
        s["accepted"] += do_accept
        s["interval_index"] += 2 * is_frozen

        s["log_prob"] = jnp.where(
            do_accept.reshape(-1), proposal_log_prob, s["log_prob"]
        )

        accepted = accepted + s["accepted"]
        delta_steps = delta_steps_val

        finished_chain = ~(s["interval_index"] < 2 * interval_count)

        update_interval_index = (s["interval_index"] + 1) % 2
        interval_index = jnp.where(finished_chain, update_interval_index, s["interval_index"])

        new_state = state.replace(
            rng=key,
            σ=s["σ"],
            log_prob=s["log_prob"],
            n_accepted_proc=accepted,
            interval_index=interval_index,
            n_steps_proc=state.n_steps_proc + delta_steps,  # + (1 + self.sweep_size) * self.n_batches
            lut_values=s["lut_values"] if self.dyn_tol_enabled else state.lut_values,
            lut_counts=s["lut_counts"] if self.dyn_tol_enabled else state.lut_counts,
            lut_last_update_step=s["last_update"] if self.dyn_tol_enabled else state.lut_last_update_step,
            lut_update_count=s["lut_update_count"] if self.dyn_tol_enabled else state.lut_update_count,
            freeze_correction=new_correction,
        )

        if self.global_proposal:
            new_state = self._propose_global_update(
                machine, parameters, new_state, key2, can_accept=finished_chain
            )

        if debug:
            return new_state, (new_state.σ, new_state.log_prob,), s

        return new_state, (new_state.σ, new_state.log_prob,)


    @partial(jax.jit, static_argnums=(1, 4))
    def _sample_typewriter_simple(self, machine, parameters, state, debug):
        """Simplified typewriter sampler without cached partial evaluations."""
        assert self.chunk_size is None, "Chunk size not implemented yet"

        apply_machine_full = self.apply_full

        key = state.rng
        σ_current = state.σ
        accepted = state.n_accepted_proc
        interval_index = state.interval_index

        interval_width = self._compute_interval_width(use_interval_mult=False)
        assert (
            σ_current.shape[-1] % interval_width == 0
        ), f"Hilbert space size {σ_current.shape[-1]} must be divisible by interval width {interval_width}"

        interval_count = σ_current.shape[-1] // interval_width // 2
        batch_size = σ_current.shape[0]

        # Always refresh the log probability with a full evaluation.
        logp_current = self.machine_pow * apply_machine_full(parameters, σ_current).real
        _assert_good_log_prob_shape(logp_current, self.n_batches, machine)

        key, key_centers = jax.random.split(key, 2)
        relative_centers = jax.random.randint(
            key_centers, (interval_count, batch_size), 0, interval_width
        )

        offset_dtype = interval_index.dtype
        offsets = jnp.arange(0, 2 * interval_count, 2, dtype=offset_dtype)
        center_offset_index = interval_index[None, :] + offsets[:, None]
        centers = relative_centers + interval_width * center_offset_index

        carry = {
            "key": key,
            "σ": σ_current,
            "log_prob": logp_current,
            "accepted": jnp.zeros_like(accepted),
            "interval_index": interval_index,
            "accepted_history": jnp.zeros((interval_count, batch_size), dtype=jnp.bool_),
        }

        def loop_body(i, s):
            key_uniform, key_move, next_key = jax.random.split(s["key"], 3)
            centers_i = centers[i]
            σp = self.local_move_rule.apply(s["σ"], centers_i, key_move)

            proposal_log_prob = self.machine_pow * apply_machine_full(parameters, σp).real
            uniform = jax.random.uniform(key_uniform, shape=(batch_size,))

            log_prob_diff = proposal_log_prob - s["log_prob"]
            accept_prob = jnp.minimum(1.0, jnp.exp(log_prob_diff))
            do_accept = uniform < accept_prob

            σ_new = jnp.where(do_accept[:, None], σp, s["σ"])
            log_prob_new = jnp.where(do_accept, proposal_log_prob, s["log_prob"])
            accepted_new = s["accepted"] + do_accept.astype(s["accepted"].dtype)
            interval_new = s["interval_index"] + 2

            accepted_history = s["accepted_history"].at[i].set(do_accept)

            return {
                "key": next_key,
                "σ": σ_new,
                "log_prob": log_prob_new,
                "accepted": accepted_new,
                "interval_index": interval_new,
                "accepted_history": accepted_history,
            }

        carry = jax.lax.fori_loop(0, interval_count, loop_body, carry)
        key_out, key_aux = jax.random.split(carry["key"], 2)

        total_accepted = accepted + carry["accepted"]
        steps_increment = jnp.asarray(
            interval_count * self.n_batches, dtype=state.n_steps_proc.dtype
        )
        interval_index_out = jnp.mod(carry["interval_index"] + 1, 2)

        new_state = state.replace(
            rng=key_out,
            σ=carry["σ"],
            log_prob=carry["log_prob"],
            n_accepted_proc=total_accepted,
            interval_index=interval_index_out,
            n_steps_proc=state.n_steps_proc + steps_increment,
            freeze_correction=jnp.ones_like(state.freeze_correction),
        )

        if self.global_proposal:
            new_state = self._propose_global_update(
                machine, parameters, new_state, key_aux
            )

        if debug:
            debug_data = {
                "centers": centers,
                "accepted_history": carry["accepted_history"],
            }
            return new_state, (new_state.σ, new_state.log_prob), debug_data

        return new_state, (new_state.σ, new_state.log_prob)

    @partial(jax.jit, static_argnums=(1, 4))
    def _sample_typewriter_exact(self, machine, parameters, state, debug):
        """
        Implementation of `sample_next` for subclasses of `MetropolisSampler`.

        If you subclass `MetropolisSampler`, you should override this and not `sample_next`
        itself, because `sample_next` contains some common logic.
        """
        assert self.chunk_size is None, "Chunk size not implemented yet"

        # full eval → (vector, cache)
        apply_machine_full = self.apply_full_headless

        # unpack incoming state
        key       = state.rng
        σ_current = state.σ
        logp_current = state.log_prob
        accepted = state.n_accepted_proc
        interval_index = state.interval_index
        interval_width = self._compute_interval_width(use_interval_mult=False)

        assert σ_current.shape[-1] % interval_width == 0, f"Hilbert space size {σ_current.shape[-1]} must be divisible by interval width {interval_width}"
        interval_count = σ_current.shape[1] // interval_width // 2
        batch_size = σ_current.shape[0]

        # full machine evaluation
        vector, cache = apply_machine_full(parameters, σ_current)
        #cache = jax.tree.map(_cast_to_bf16, cache)

        # extract log‐prob
        logp_current = self.machine_pow * logcosh_activation(vector).real

        paramDict = {"params": parameters["params"], **cache}

        # Generate the center arrays 
        key, key_centers, key_moves_base = jax.random.split(key, 3)
        relative_centers = jax.random.randint(key_centers, (interval_count, batch_size,), 0, interval_width)
        center_offset_index = interval_index[None, :] + jnp.arange(0, 2 * interval_count, 2)[:, None] 
        centers = relative_centers + interval_width * center_offset_index
        key_moves = jax.random.split(key_moves_base, interval_count)

        # This function slices the array 
        def _one_sweep(σ, centers, key_move):
            """
            All computations that depended only on (key1, key2, σ_current)
            are pure and can therefore be vmapped.
            """
            idx           = jnp.arange(batch_size, dtype=jnp.int32)
            σp = self.local_move_rule.apply(σ, centers, key_move)
        
            σp_sliced     = slice_tokens_jit(
                σp, idx, centers, width=self.width, token_size=self.token_size
            )
            return σp_sliced

        # Broadcast σ_current once; map over the key axes (axis-0)
        σ_prop_chain_sliced  = jax.vmap(_one_sweep, in_axes=(None, 0, 0))(
            σ_current, centers, key_moves
        )

        idx = jnp.arange(batch_size, dtype=jnp.int32)
        token_centers_flat = centers // self.token_size 

        # Apply the actual network 
        apply_func = jax.vmap(
            lambda p, s, id, c: self.apply_headless_partial(p, s, centers=(id, c)), in_axes=(None, 0, None, 0), out_axes=0
        )
        vectorsNewFull = apply_func(paramDict, σ_prop_chain_sliced, idx, token_centers_flat)
        vectorsNew = vectorsNewFull - vector[None, :]  # Subtract the base vector
        
        # Prepare the carry for history replay
        carry_replay = {
            "key": key,
            "σ": σ_current,
            "σ_prop": σ_current,
            "proposal_centers" : centers,
            "log_prob": logp_current,
            "freeze_iter": jnp.full_like(logp_current, interval_count, dtype=jnp.int32), 
            "prev_log_prob": logp_current,
            "accepted": jnp.zeros_like(accepted, dtype=jnp.int32),
            "replay_vectors": vectorsNew,
            "replay_vectors_debug" : vectorsNewFull, # Debugging purposes
            "current_vector": vector,
            "proposal_accepted": jnp.zeros_like(centers, dtype=jnp.int32),
            "interval_index" : interval_index, 
            "stored_u" : jnp.ones_like(logp_current), 
            "proposal_keys": key_moves,
        }

        # ─── Remaing Metropolis steps ───
        def loop_history_replay(i, s):
            # Make new key for this iteration
            key2, next_key = jax.random.split(s["key"], 2) 
            s["key"] = next_key  # Update the stored key
            interval_index = s["interval_index"]

            centers = s["proposal_centers"][i] 
            key_move = s["proposal_keys"][i]
            σp = self.local_move_rule.apply(s["σ"], centers, key_move)

            prev_log_prob = s["prev_log_prob"]
            # Freeze markov chains that collided with previous updates and save the proposal
            vectorNew = s["replay_vectors"][i] # This is the vector that was computed before

            proposal_log_prob_approx = self.machine_pow * logcosh_activation(s["current_vector"] + vectorNew).real
            
            wavefunc_true = self.apply_full(parameters, σp)
            proposal_log_prob = self.machine_pow * wavefunc_true

            log_prob_correction = None 
            assert log_prob_correction is None, "logprob correction not implemented yet"
            if log_prob_correction is not None:
                metropolis_frac = jnp.exp(
                    proposal_log_prob - s["log_prob"] + log_prob_correction
                )
            else:
                metropolis_frac = jnp.exp(proposal_log_prob - s["log_prob"]) 
                metropolis_frac_approx = jnp.exp(proposal_log_prob_approx - s["log_prob"])

            metropolis_frac = jnp.clip(metropolis_frac, 0.0, 1.0)
            metropolis_frac_approx = jnp.clip(metropolis_frac_approx, 0.0, 1.0)
            uniform = jax.random.uniform(key2, shape=(batch_size,))

            if self.dyn_tol_enabled and state.lut_values is not None:
                tau = self._lut_query(state, s["accepted"]).astype(metropolis_frac.dtype)
            else:
                tau = self.accept_tol

            is_close = (jnp.abs(uniform - metropolis_frac_approx) < tau) & (s["log_prob"] < jnp.inf)
            in_chain = interval_index < 2 * interval_count
            is_close = jnp.logical_and(is_close, in_chain) # Can only freeze inside the chain 
            


            s["σ_prop"] = jnp.where(is_close[:, None], σp, s["σ_prop"])
            
            s["freeze_iter"] = jnp.where(is_close, i, s["freeze_iter"])
            #s["log_prob_correction"] = jnp.where(collides_all & (s["log_prob"] > -jnp.inf), log_prob_correction, s["log_prob"])
            s["prev_log_prob"] = jnp.where(is_close, s["log_prob"], prev_log_prob)
            s["log_prob"] = jnp.where(is_close, jnp.inf, s["log_prob"]) # Set log_prob to -inf for collided vectors

            do_accept = (uniform < metropolis_frac) & ~is_close
            do_accept = jnp.logical_and(do_accept, in_chain) # Can only accept if end of chain not reached
        
            s["proposal_accepted"] = s["proposal_accepted"].at[i].set(do_accept)
            s["current_vector"] = jnp.where(do_accept.reshape(-1, 1), s["current_vector"] + vectorNew, s["current_vector"])
            
            # do_accept must match ndim of proposal and state (which is 2)
            s["σ"] = jnp.where(do_accept.reshape(-1, 1), σp, s["σ"])
            s["accepted"] += do_accept
            
            # Advance the interval index only for the unfrozen chains 
            unfrozen = (s["log_prob"] < jnp.inf) & ~is_close & in_chain
            s["interval_index"] += 2*unfrozen 
            s["stored_u"] = jnp.where(is_close, uniform, s.get("stored_u", jnp.ones_like(uniform)))

            s["log_prob"] = jnp.where(
                do_accept.reshape(-1), proposal_log_prob, s["log_prob"]
            )
            return s

        s = jax.lax.fori_loop(0, interval_count, loop_history_replay, carry_replay)
        key, key2 = jax.random.split(s["key"], 2)
        
        # Propose the updates, propose idenical state for unfrozen chains 
        proposal = jnp.where(
            (s["log_prob"] < jnp.inf)[:, None], s["σ"], s["σ_prop"]
        )
        prev_probability = jnp.where(
            s["log_prob"] < jnp.inf,  s["log_prob"], s["prev_log_prob"]
        )

        prob = self.apply_full(parameters, proposal)
        proposal_log_prob = self.machine_pow * prob

        #_assert_good_log_prob_shape(proposal_log_prob, self.n_batches, machine)
        uniform = s["stored_u"]
        log_prob_correction =  None #s["log_prob_correction"]
    
        if log_prob_correction is not None:
            do_accept = uniform < jnp.exp(
                proposal_log_prob - prev_probability + log_prob_correction
            )
        else:
            do_accept = uniform < jnp.exp(proposal_log_prob - prev_probability) 

        # Update only the ones that are inside the chain 
        in_chain = s["interval_index"] < 2 * interval_count
        do_accept = jnp.logical_and(do_accept, in_chain)

        # Update the accepted vectors
        s["σ"] = jnp.where(do_accept.reshape(-1, 1), proposal, s["σ"])
        freeze_iter = s["freeze_iter"]

        prop_accepted_update = jnp.where(freeze_iter < interval_count, do_accept, s["proposal_accepted"][freeze_iter, idx])
        s["proposal_accepted"] = s["proposal_accepted"].at[freeze_iter, idx].set(prop_accepted_update)
        
        frozen = s["freeze_iter"] < interval_count
        s["interval_index"] += 2*frozen
        
        s["accepted"] += do_accept

        s["log_prob"] = jnp.where(
            do_accept.reshape(-1), proposal_log_prob, prev_probability
        )

        accepted = accepted + s["accepted"]
        delta_steps = jnp.sum(s["freeze_iter"])
        delta_steps_float = delta_steps.astype(jnp.float32)
        batch_size_f = jnp.asarray(batch_size, dtype=jnp.float32)
        mean_freeze = delta_steps_float / jnp.maximum(batch_size_f, 1.0)
        interval_count_f = jnp.asarray(interval_count, dtype=jnp.float32)
        new_correction = interval_count_f / jnp.maximum(mean_freeze, 1e-6)
        new_correction = jnp.maximum(new_correction, 1.0)

        finished_chain = ~(s["interval_index"] < 2 * interval_count)

        update_interval_index = (s["interval_index"] + 1)%2
        interval_index = jnp.where(finished_chain, update_interval_index, s["interval_index"])

        new_state = state.replace(
            rng=key,
            σ=s["σ"],
            log_prob=s["log_prob"],
            n_accepted_proc= accepted,
            interval_index=interval_index,
            n_steps_proc=state.n_steps_proc + delta_steps, # + (1 + self.sweep_size) * self.n_batches,
            freeze_correction=new_correction,
        )

        if self.global_proposal: 
            new_state =  self._propose_global_update(
                machine, parameters, new_state, key, can_accept=finished_chain
            )

        if debug:
            return new_state, (new_state.σ, new_state.log_prob, ), s

        return new_state, (new_state.σ, new_state.log_prob, )


    @partial(jax.jit, static_argnames=("machine", ))
    def _sample_typewriter_sweep(self, machine, parameters, state):
        active_sweep_size = jnp.asarray(
            state.typewriter_target_steps, dtype=jnp.int32
        )
        active_sweep_size = jnp.maximum(
            active_sweep_size, jnp.asarray(1, dtype=jnp.int32)
        )

        interval_width = self._compute_interval_width(use_interval_mult=True)
        eff_sweep_size = (
            2 * interval_width * active_sweep_size // self.hilbert.size
        )
        eff_sweep_size = jnp.maximum(
            eff_sweep_size, jnp.asarray(1, dtype=jnp.int32)
        )

        sample_typewriter = self._sample_typewriter
        if self.shift_project:
            sample_typewriter = self._sample_typewriter_shift_project
        elif self.simple_typewriter:
            sample_typewriter = self._sample_typewriter_simple

        if self.dyn_tol_enabled:
            target_steps = jnp.ceil(eff_sweep_size * state.freeze_correction).astype(jnp.int32)
            target_steps = jnp.maximum(target_steps, jnp.asarray(1, dtype=jnp.int32))
        else:
            target_steps = jnp.asarray(eff_sweep_size, dtype=jnp.int32)
            target_steps = jnp.maximum(target_steps, jnp.asarray(1, dtype=jnp.int32))

        def cond_fun(loop_state):
            step_idx, max_steps, _ = loop_state
            return step_idx < max_steps

        def body_fun(loop_state):
            step_idx, max_steps, old_state = loop_state
            new_state, _ = sample_typewriter(
                machine, parameters, old_state, False
            )
            return (step_idx + 1, max_steps, new_state)

        init_loop = (
            jnp.asarray(0, dtype=jnp.int32),
            target_steps,
            state,
        )
        _, _, state = jax.lax.while_loop(cond_fun, body_fun, init_loop)

        return state, (state.σ, state.log_prob)


    @partial(
        jax.jit, static_argnames=("machine", "chain_length", "return_log_probabilities")
    )
    def _sample_chain(
        self,
        machine,
        parameters,
        state,
        chain_length,
        return_log_probabilities: bool = False,
    ):
        """
        Samples `chain_length` batches of samples along the chains.

        Internal method used for jitting calls.

        Arguments:
            machine: A Flax module with the forward pass of the log-pdf.
            parameters: The PyTree of parameters of the model.
            state: The current state of the sampler.
            chain_length: The length of the chains.

        Returns:
            σ: The next batch of samples.
            state: The new state of the sampler
        """

        # if self.sample_fast:
        #     scan_fun = lambda state, _: self._sample_next_super_fast_auto_stop(
        #         machine, parameters, state, debug=False
        #     )
        # else:
        #     scan_fun = lambda state, _: self._sample_next(machine, parameters, state)

        scan_fun = lambda state, _, : self._sample_typewriter_sweep(machine, parameters, state) 

        state, (samples, log_probabilities) = jax.lax.scan(
            scan_fun,
            state,
            xs=None,
            length=chain_length,
        )
        # make it (n_chains, n_samples_per_chain) as expected by netket.stats.statistics
        samples = jnp.swapaxes(samples, 0, 1)
        log_probabilities = jnp.swapaxes(log_probabilities, 0, 1)

        if return_log_probabilities:
            return (samples, log_probabilities), state
        else:
            return samples, state



    @partial(jax.jit, static_argnums=(0, 1))        # only `self` is static
    def _propose_global_update(self, machine, parameters, state, key, can_accept = None):
        """
        Try a *global* spin flip (σ -> -σ) on every Markov chain in the batch.

        Parameters
        ----------
        machine      : nn.Module | Flax transform
            Same object that is passed around in the other sampler helpers.
            Needed only for the shape-check helper.
        parameters   : FrozenDict
            Network parameters.
        state        : SamplerState  (same type returned by `sample_next`)
        key          : jax.random.PRNGKey
            Source of randomness for the Metropolis test.
        debug        : bool
            If True, also return the acceptance mask.

        Returns
        -------
        new_state    : SamplerState
            State after the Metropolis step.
        accepted     : jax.Array[bool]   (only if debug=True)
            Per-chain acceptance mask.
        """
        # ------------------------------------------------------------------
        # 1. Current configuration & probability
        # ------------------------------------------------------------------
        σ_current   = state.σ                        # (n_chains, L)
        logp_current = state.log_prob                # (n_chains,)

        # ------------------------------------------------------------------
        # 2. Proposal : *global* flip
        # ------------------------------------------------------------------
        σ_proposal = -σ_current

        # single full forward pass (identical to the very end of
        # `_sample_next_super_fast_auto_stop`)
        vector_prop = self.apply_full(parameters, σ_proposal)
        logp_prop   = self.machine_pow * vector_prop.real      # (n_chains,)

        # sanity-check, re-using the helper already present in the code base
        _assert_good_log_prob_shape(logp_prop, self.n_batches, machine)

        # ------------------------------------------------------------------
        # 3. Metropolis test
        # ------------------------------------------------------------------
        key, key_u = jax.random.split(key)                     # keep one key for the new state
        uniform    = jax.random.uniform(key_u, shape=(self.n_batches,))

        accept_prob = jnp.exp(logp_prop - logp_current)        # detailed balance
        accept_prob = jnp.minimum(1.0, accept_prob)            # clip for numerical safety
        do_accept   = uniform < accept_prob                    # (n_chains,)

        if can_accept is not None:
            # If can_accept is provided, we only accept the proposal if it is True
            do_accept = jnp.logical_and(do_accept, can_accept)
        # ------------------------------------------------------------------
        # 4. Assemble the new state
        # ------------------------------------------------------------------
        σ_new   = jnp.where(do_accept[:, None], σ_proposal, σ_current)
        logp_new = jnp.where(do_accept,           logp_prop,   logp_current)
        acc_new  = state.n_accepted_proc + do_accept           # keep per-chain counter

        new_state = state.replace(
            rng            = key,
            σ              = σ_new,
            log_prob       = logp_new,
            n_accepted_proc= acc_new,
            n_steps_proc   = state.n_steps_proc + self.n_batches  # one sweep for every chain
        )

        return new_state


    def check_for_mismatched_chains(self, sigma, machine, params, typewriter=False):
        if typewriter: 
            sample_fun = self._sample_typewriter_shift_project if self.shift_project else self._sample_typewriter
            _, (σ, _), s = sample_fun(machine, params, self.reset(machine, params), True, True)
            _, _, s_exact = self._sample_typewriter_exact(machine, params, self.reset(machine, params), True, True)
        else: 
            (new_state, σ, logp_fast, s), (new_state_exact, σ_exact, logp_exact, s_exact) = self.compare_samplers(sigma, machine, params)
        # Function to calculate distance with periodic boundary conditions
        
        L = σ.shape[-1]  # Assuming σ is a 2D array with shape (n_chains, L)
        def pbc_distance(center1, center2, L):
            diff = abs(center1 - center2)
            return min(diff, L - diff)

        # Get the freeze iteration for each chain
        freeze_iters = np.array(s["freeze_iter"])  # shape (n_chains,)
        n_chains = freeze_iters.shape[0]

        # Get the data for analysis
        centers      = np.array(s["proposal_centers"])   # (n_steps, n_chains)
        accepted     = np.array(s["proposal_accepted"])
        centers_exact = np.array(s_exact["proposal_centers"])
        accepted_exact = np.array(s_exact["proposal_accepted"])

        # Find chains with a first error before the freeze iteration
        error_chains = []
        for chain_idx in range(n_chains):
            fast_chain  = accepted[:, chain_idx]
            exact_chain = accepted_exact[:, chain_idx]
            diffs       = fast_chain != exact_chain
            if np.any(diffs):
                first_err = np.argmax(diffs)  # first True index
                # only keep if it happens before freeze
                if first_err < freeze_iters[chain_idx]:
                    error_chains.append((chain_idx, first_err))

        print(f"Found {len(error_chains)} chains with errors before freeze")

        # Compute per‐error distances
        error_mean_distances = []
        error_min_distances  = []

        for chain_idx, err_step in error_chains:
            err_center = centers[err_step, chain_idx]
            prev_centers = centers[:err_step, chain_idx]
            prev_acc     = accepted[:err_step, chain_idx]
            acc_centers  = prev_centers[prev_acc == 1]

            if acc_centers.size:
                dists = [pbc_distance(err_center, pc, L) for pc in acc_centers]
                m    = np.mean(dists)
                mn   = np.min(dists)
                error_mean_distances.append(m)
                error_min_distances.append(mn)
                print(f"Chain {chain_idx}, step {err_step}: "
                    f"{len(acc_centers)} prev accepted, mean={m:.2f}, min={mn:.2f}")
            else:
                print(f"Chain {chain_idx}, step {err_step}: no previous accepted centers")

        if error_mean_distances:
            print("\nOverall stats:")
            print(f"Mean distance: {np.mean(error_mean_distances):.2f} ± {np.std(error_mean_distances):.2f}")
            print(f"Min distance:  {np.mean(error_min_distances):.2f} ± {np.std(error_min_distances):.2f}")
        else:
            print("No valid error moves to analyze")
            return 


        # Histogram of minimum distances at error moves
        plt.figure(figsize=(8, 5))
        bins = np.arange(0, int(np.max(error_min_distances)) + 2)  # integer bins
        plt.hist(error_min_distances, bins=bins, color='skyblue', edgecolor='black', alpha=0.8)
        plt.xlabel("Minimum Distance to Previous Accepted Centers")
        plt.ylabel("Count of Error Moves")
        plt.title("Histogram of Min Distances for Error Moves")
        plt.grid(alpha=0.3)
        plt.show()

    def check_typerwriter_mismatches(self, sigma, machine, params, iterations = 10):
        """
        Checks for mismatches in the typewriter sampler.
        """
        # Initialize lists to store sigma arrays
        error_count = []
        for j in tqdm(range(iterations)):
            
            state = self.reset(machine, params)
            log_prob = self.apply_full(params, sigma)
            state = state.replace(σ=sigma, log_prob=log_prob)
            sample_fun = self._sample_typewriter_shift_project if self.shift_project else self._sample_typewriter
            a, b, s = sample_fun(machine, params, state, True)
            a, b, s_exact = self._sample_typewriter_exact(machine, params, state, True, True)
            
            
            freeze_iter = jnp.minimum(s["freeze_iter"], s_exact["freeze_iter"])
            difference = [s_exact["proposal_accepted"][:, b][:freeze_iter[b]]- s["proposal_accepted"][:, b][:freeze_iter[b]] for b in range(1008)]
            correct = jnp.array([jnp.allclose(d,0) for d in difference])
            first_deviation = jnp.array([jnp.argmax(~d)  for d in difference if jnp.any(~d)])
            

            res = jnp.sum(~correct)
            error_count.append(res)
            if res != 0:
                print(f"Found {res} mismatches in iteration {j}")
                print("First deviation at steps:", first_deviation)
                print("Proposal accepted:", s["proposal_accepted"])
                print("Exact proposal accepted:", s_exact["proposal_accepted"])

        print(f"Final concatenated shapes:")
        print(f"In total found {jnp.sum(jnp.array(error_count))} mismatches in {len(error_count)} iterations and {len(error_count)*sampler.n_chains} samples.")

    def check_freeze_iteration(self, sigma, machine, params):
        """
        Checks the freeze iteration of the sampler.
        """
        (new_state, σ_fast, logp_fast, s), (new_state_exact, σ_exact, logp_exact, s_exact) = self.compare_samplers(sigma, machine, params)


        plt.figure(figsize=(15, 6))

        # First subplot - histogram
        plt.subplot(1, 2, 1)
        freeze_values = np.array(s["freeze_iter"])
        plt.hist(freeze_values, bins=range(0, max(freeze_values) + 2), 
                alpha=0.7, edgecolor='black', color='skyblue')
        plt.xlabel('Freeze Iteration')
        plt.ylabel('Number of Chains')
        plt.title('Distribution of Freeze Iterations')
        plt.grid(True, alpha=0.3)

        # Add statistics
        plt.text(0.7, 0.8, f'Total chains: {len(freeze_values)}\n'
                        f'Mean freeze iter: {np.mean(freeze_values):.2f}\n'
                        f'Median freeze iter: {np.median(freeze_values):.2f}\n'
                        f'Max freeze iter: {np.max(freeze_values)}\n'
                        f'Min freeze iter: {np.min(freeze_values)}',
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # Second subplot - cumulative distribution
        plt.subplot(1, 2, 2)
        counts, bins = np.histogram(freeze_values, bins=range(0, max(freeze_values) + 2))
        cumulative = np.cumsum(counts) / len(freeze_values)  # Normalize to sum to 1
        bin_centers = bins[:-1]

        plt.plot(bin_centers, cumulative, 'ro-', linewidth=2, markersize=4)
        plt.xlabel('Freeze Iteration')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution of Freeze Iterations')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

        # Add horizontal lines for reference
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50%')
        plt.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='80%')
        plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='90%')
        plt.legend()

        plt.tight_layout()
        plt.show()

        print(f"Freeze iteration statistics:")
        print(f"Mean: {np.mean(freeze_values):.2f}")
        print(f"Median: {np.median(freeze_values):.2f}")
        print(f"Standard deviation: {np.std(freeze_values):.2f}")
        print(f"Range: {np.min(freeze_values)} - {np.max(freeze_values)}")
        print(f"Chains that never froze (freeze_iter = {max(freeze_values)}): {np.sum(freeze_values == max(freeze_values))}")


    def __repr__(self):
        if self.sweep_schedule is not None:
            schedule_desc = (
                self.sweep_schedule_config
                if self.sweep_schedule_config is not None
                else "<callable>"
            )
            schedule_line = f"\n  sweep_schedule = {schedule_desc},"
        else:
            schedule_line = ""

        return (
            f"{type(self).__name__}("
            + f"\n  hilbert = {self.hilbert},"
            + f"\n  rule = {self.rule},"
            + f"\n  n_chains = {self.n_chains},"
            + f"\n  sweep_size = {self.sweep_size},"
            + schedule_line
            + f"\n  reset_chains = {self.reset_chains},"
            + f"\n  machine_power = {self.machine_pow},"
            + f"\n  dtype = {self.dtype}"
            + ")"
        )

    def __str__(self):
        parts = [
            f"rule = {self.rule}",
            f"n_chains = {self.n_chains}",
            f"sweep_size = {self.sweep_size}",
        ]
        if self.sweep_schedule is not None:
            parts.append("dynamic_sweep = True")
        parts.extend(
            [
                f"reset_chains = {self.reset_chains}",
                f"machine_power = {self.machine_pow}",
                f"dtype = {self.dtype}",
            ]
        )
        return f"{type(self).__name__}(" + ", ".join(parts) + ")"


