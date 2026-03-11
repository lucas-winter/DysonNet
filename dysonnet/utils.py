import copy
import functools
import pathlib
import time
from dataclasses import dataclass, field, replace as dc_replace
from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
import numpy.typing as npt
import optax
from netket import jax as nkjax
from netket.nn import split_array_mpi
from netket.operator.spin import sigmax, sigmaz
from netket.optimizer.qgt.qgt_jacobian_common import to_shift_offset, rescale
from netket.optimizer.qgt.qgt_jacobian_pytree import QGTJacobianPyTreeT
from netket.sampler import MetropolisRule
from netket.utils.struct import dataclass as nk_dataclass

import dysonnet.DysonNQS as mnqs
import dysonnet.custom_sampler as cs
from dysonnet.DysonBlock import DysonBlock
from dysonnet.custom_operator import (
    TurboTurboOperator,
    hamiltonian_j1j2,
    hamiltonian_tfim,
    magnetization_op,
    magnetization_sqr_op,
    staggered_magnetization_op,
    staggered_magnetization_sqr_op,
    structure_factor_zz,
    zz_correlators,
)

REAL_DTYPE = jnp.asarray(1.0).dtype


def circulant(
    row: npt.ArrayLike, times: Optional[int] = None
) -> npt.ArrayLike:
    """Build a (full or partial) circulant matrix based on an array.

    Source: https://arxiv.org/html/2407.04773v1.

    Args:
        row: The first row of the matrix.
        times: If not None, the number of rows to generate.

    Returns:
        If `times` is None, a square matrix with all the offset versions of the
        first argument. Otherwise, `times` rows of a circulant matrix.
    """
    row = jnp.asarray(row)

    def scan_arg(carry, _):
        new_carry = jnp.roll(carry, -1)
        return (new_carry, new_carry)

    if times is None:
        nruter = jax.lax.scan(scan_arg, row, row)[1][::-1, :]
    else:
        nruter = jax.lax.scan(scan_arg, row, None, length=times)[1][::-1, :]

    return nruter


class BestIterKeeper:
    """Store the values of a bunch of quantities from the best iteration.

    Source: https://arxiv.org/html/2407.04773v1.

    "Best" is defined in the sense of lowest energy.

    Args:
        Hamiltonian: An array containing the Hamiltonian matrix.
        N: The number of spins in the chain.
        baseline: A lower bound for the V score. If the V score of the best
            iteration falls under this threshold, the process will be stopped
            early.
        filename: Either None or a file to write the best state to.
    """

    def __init__(
        self,
        Hamiltonian: npt.ArrayLike,
        N: int,
        baseline: float,
        filename: Optional[pathlib.Path] = None,
    ):
        """Initialize the keeper.

        Source: https://arxiv.org/html/2407.04773v1.
        """
        self.Hamiltonian = Hamiltonian
        self.N = N
        self.baseline = baseline
        self.filename = filename
        self.vscore = np.inf
        self.best_energy = np.inf
        self.best_state = None

    def update(self, step, log_data, driver):
        """Update the stored quantities if necessary.

        Source: https://arxiv.org/html/2407.04773v1.

        This function is intended to act as a callback for NetKet. Please refer
        to its API documentation for a detailed explanation.
        """
        vstate = driver.state
        energystep = np.real(vstate.expect(self.Hamiltonian).mean)
        var = np.real(getattr(log_data[driver._loss_name], "variance"))
        mean = np.real(getattr(log_data[driver._loss_name], "mean"))
        varstep = var / energystep**2

        if energystep < self.best_energy and energystep > -jnp.inf:
            self.best_energy = energystep
            self.best_state = copy.copy(driver.state)
            self.best_state.parameters = flax.core.copy(
                driver.state.parameters
            )
            self.vscore = varstep

            if self.filename != None:
                with open(self.filename, "wb") as file:
                    file.write(flax.serialization.to_bytes(driver.state))

        return self.vscore > self.baseline


@nk_dataclass
class InvertMagnetization(MetropolisRule):
    """Monte Carlo mutation rule that inverts all the spins.

    Source: https://arxiv.org/html/2407.04773v1.

    Please refer to the NetKet API documentation for a detailed explanation of
    the MetropolisRule interface.
    """

    def transition(rule, sampler, machine, parameters, state, key, σ):
        """Apply the inversion transition.

        Source: https://arxiv.org/html/2407.04773v1.
        """
        indxs = jax.random.randint(
            key, shape=(1,), minval=0, maxval=sampler.n_chains
        )
        σp = σ.at[indxs, :].multiply(-1)
        return σp, None


def get_Hamiltonian(
    N: int,
    J: float,
    alpha: float,
    trans_field: float = -1.0,
    sym_field: Optional[bool] = False,
    epsilon: Optional[float] = 1e-3,
    return_norm: Optional[float] = False,
) -> Tuple[npt.ArrayLike, float]:
    """Build the Hamiltonian of the system.

    Source: https://arxiv.org/html/2407.04773v1.

    Args:
        N: The number of spins in the chain.
        J: The bare spin-spin interaction strength.
        alpha: The exponent of the power-law governing the decay of the
            interaction strength.
        trans_field: The transverse component of an external field.
        sym_field: The longitudinal component of an external field.
            It lifts the degeneracy in the ordered phase breaking the symmetry.
        epsilon: The relative strength of the longitudinal component respect
            to the interaction strength.
        return_norm: Flag that allows the function to return the value of the
            normalization constant.

    Returns:
        Either H or the tuple (H, N_norm) depending on whether return_norm
        is True.
            H: Array containing the Hamiltonian matrix.
            N_norm: Float value of the normalization constant.
    """

    hi = nk.hilbert.Spin(s=1 / 2, N=N)
    H = sum(trans_field * sigmax(hi, i) for i in range(N))

    N_norm = 1
    for i in range(1, N):
        dist = min(abs(i), N - abs(i))
        N_norm += 1 / dist**alpha

    J = J / N_norm

    for i in range(0, N):
        for j in range(i, N):
            dist = min(abs(i - j), N - abs(i - j))
            cn = 1.0
            if dist == 0:
                dist = 1
                cn = 2.0
            H += J / cn * sigmaz(hi, i) * sigmaz(hi, j) / (dist**alpha)

    if sym_field:
        H += J * epsilon * sum(sigmaz(hi, i) for i in range(N))

    H /= N
    if return_norm:
        return (H, N_norm)
    else:
        return H


def get_eigvals(
    Hamiltonian: npt.ArrayLike, order: int = 1, eigenvecs: bool = False
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """Partially diagonalize a given matrix, namely the Hamiltonian.

    Source: https://arxiv.org/html/2407.04773v1.

    Args:
        Hamiltonian: Hamiltonian or general matrix to diagonalize.
        order: The number of lowest energy levels that we want to obtain.
        eigenvecs: If set to True, returns also an Array with the eigenvectors
            of the corresponding eigenlevels obtained.

    Returns:
        Either w or the tuple (w, v) depending on whether compute_eigenvectors
        is True.
            w: Array containing the lowest 'order' eigenvalues.
            v: Array containing the eigenvectors as columns, such that
                'v[:, i]' corresponds to w[i].
    """

    return nk.exact.lanczos_ed(
        Hamiltonian, k=order, compute_eigenvectors=eigenvecs
    )

def force_link_tensor_true(obj):
    # 1) You only have the CLASS (not constructed yet)
    if isinstance(obj, type) and issubclass(obj, nn.Module):
        # bake it into the constructor
        return obj.partial(link_tensor_approximate=True)

    # 2) You have an UNBOUND INSTANCE (constructed, not bound)
    if isinstance(obj, nn.Module):
        # avoid .replace(); use dataclasses.replace
        return dc_replace(obj, link_tensor_approximate=True)

    # 3) You have a BOUND MODULE (e.g., from model.bind(...))
    # BoundModule doesn't have .replace; unbind → replace → rebind
    if hasattr(obj, "unbind") and hasattr(obj, "variables"):
        bound = obj
        unbound = bound.unbind()  # returns the original nn.Module
        new_unbound = dc_replace(unbound, link_tensor_approximate=True)
        return new_unbound.bind(bound.variables)

    # 4) You only control apply() (e.g., third-party module)
    # Pin the call arg at apply-time as a last-resort:
    if callable(obj):
        return functools.partial(obj, link_tensor_approximate=True)

    raise TypeError("Unsupported object passed to force_link_tensor_true")


def make_linear_sweep_ramp(
    total_iters: int,
    *,
    start_sweeps: int | None = None,
    end_sweeps: int | None = None,
    start_multiplier: float | None = None,
    end_multiplier: float | None = None,
    min_sweeps: int = 1,
    max_sweeps: int | None = None,
) -> cs.SweepRampConfig:
    """
    Convenience helper to build a linear sweep-size schedule.

    Args:
        total_iters: Number of optimisation iterations over which to ramp.
        start_sweeps: Absolute sweep count used at iteration 0. Defaults to the sampler's base sweep size.
        end_sweeps: Absolute sweep count reached at the end of the ramp. Defaults to the base sweep size.
        start_multiplier: Multiplier applied to the base sweep size when `start_sweeps` is not provided.
        end_multiplier: Multiplier applied to the base sweep size when `end_sweeps` is not provided.
        min_sweeps: Minimum sweep count enforced throughout the schedule.
        max_sweeps: Optional cap on the sweep count.
    """
    return cs.SweepRampConfig(
        plateau_iters=0,
        ramp_iters=total_iters,
        start_sweeps=start_sweeps,
        end_sweeps=end_sweeps,
        start_multiplier=start_multiplier,
        end_multiplier=end_multiplier,
        min_sweeps=min_sweeps,
        max_sweeps=max_sweeps,
    )


def make_hockey_sweep_ramp(
    plateau_iters: int,
    ramp_iters: int,
    *,
    plateau_sweeps: int | None = None,
    final_sweeps: int | None = None,
    plateau_multiplier: float | None = None,
    final_multiplier: float | None = None,
    min_sweeps: int = 1,
    max_sweeps: int | None = None,
) -> cs.SweepRampConfig:
    """
    Build a "hockey-stick" schedule that stays flat for `plateau_iters` iterations
    before ramping linearly for `ramp_iters` iterations.
    """
    return cs.SweepRampConfig(
        plateau_iters=plateau_iters,
        ramp_iters=ramp_iters,
        start_sweeps=plateau_sweeps,
        end_sweeps=final_sweeps,
        start_multiplier=plateau_multiplier,
        end_multiplier=final_multiplier,
        min_sweeps=min_sweeps,
        max_sweeps=max_sweeps,
    )


def _acf_fft(x: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """
    Fast autocorrelation function via FFT for a 1D array.
    Returns ACF from lag 0..max_lag (inclusive), normalized so acf[0] = 1.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if max_lag is None or max_lag >= n:
        max_lag = n - 1
    x = x - np.mean(x)
    if np.allclose(x, 0):
        # Constant series: define acf as 1 at lag 0 and 0 elsewhere
        acf = np.zeros(max_lag + 1, dtype=float)
        acf[0] = 1.0
        return acf

    # Next power of 2 for zero-padding
    nfft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n=nfft)
    sxx = fx * np.conjugate(fx)
    acov = np.fft.irfft(sxx, n=nfft)[:n]
    # Unbiased normalization by number of pairs at each lag
    acov = acov / np.arange(n, 0, -1)
    acf_full = acov / acov[0]
    return acf_full[: max_lag + 1]


def _integrated_time_emcee(x: np.ndarray, c: float = 5.0, max_lag: int | None = None) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        return float(n)
    if max_lag is None:
        max_lag = n - 1

    acf = _acf_fft(x, max_lag=max_lag)

    # Hard truncate at first negative (keeps estimator stable)
    first_neg = np.argmax(acf[1:] < 0) + 1
    if first_neg > 1:
        acf[first_neg:] = 0.0

    tau_old = np.inf
    for _ in range(10):
        m = np.argmax(acf[1:] < 0) + 1
        if m <= 1:
            m = min(max_lag, n - 1)
        tau = 1 + 2 * np.sum(acf[1:m])
        m = int(min(max_lag, max(1, int(c * tau))))
        tau = 1 + 2 * np.sum(acf[1 : min(m, len(acf))])
        if abs(tau - tau_old) / tau < 1e-2:
            break
        tau_old = tau

    # Don’t silently clip to n; return tau as estimated
    return float(tau)



def _split_rhat(chains: np.ndarray) -> float:
    """
    Split-R̂ (Gelman-Rubin) for multiple chains with optional splitting.
    Input: chains shape (n_chains, n_draws).
    Returns sqrt(Var^+ / W). Uses standard (non-rank) split-Rhat.
    """
    x = np.asarray(chains, dtype=float)
    C, N = x.shape
    if C < 2 or N < 4:
        return np.nan

    # Split each chain into two halves
    half = N // 2
    if half < 2:
        return np.nan
    xs = np.concatenate([x[:, :half], x[:, -half:]], axis=0)  # (2C, half)
    m, n = xs.shape
    chain_means = np.mean(xs, axis=1)
    chain_vars = np.var(xs, axis=1, ddof=1)

    W = np.mean(chain_vars)
    B = n * np.var(chain_means, ddof=1)
    # marginal posterior variance estimate
    var_plus = (n - 1) / n * W + B / n
    if W <= 0:
        return np.nan
    rhat = np.sqrt(var_plus / W)
    return float(rhat)


def _rhat_trace(chains: np.ndarray, start: int = 20, step: int = 1) -> np.ndarray:
    """
    Compute split-Rhat for prefixes chains[:, :t] for t=start..T with step.
    Returns an array aligned with t indices (same length as range).
    """
    C, T = chains.shape
    times = range(start, T + 1, step)
    vals = []
    for t in times:
        vals.append(_split_rhat(chains[:, :t]))
    return np.array(vals)


def _geweke_burn_in(x: np.ndarray, z_thresh: float = 2.0) -> int:
    """
    Geweke diagnostic-based burn-in for a single chain:
    Compare the first 10% with the last 50% using spectral variance estimates.
    We slide the start point forward until |z| < z_thresh.
    Returns the earliest index (0-based) after which the chain looks stationary under this test.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 50:
        return n // 2

    def spectral_var(y: np.ndarray) -> float:
        # Newey-West style lag truncation at floor(n**(1/3))
        acf = _acf_fft(y, max_lag=len(y) - 1)
        L = int(np.floor(len(y) ** (1 / 3)))
        w = 1 - np.arange(1, L + 1) / (L + 1)
        var = np.var(y, ddof=1)
        s = var + 2 * np.sum(w * var * acf[1 : L + 1])
        return max(s, 1e-12)

    last = x[int(0.5 * n) :]
    mean_last = np.mean(last)
    sv_last = spectral_var(last)

    # slide start from 0 until z within threshold
    for start in range(0, int(0.5 * n)):
        first_end = max(int(0.1 * (n - start)), 10)
        first = x[start : start + first_end]
        if len(first) < 10:
            break
        sv_first = spectral_var(first)
        z = (np.mean(first) - mean_last) / np.sqrt(sv_first / len(first) + sv_last / len(last))
        if np.isfinite(z) and abs(z) < z_thresh:
            return start
    return n // 2


def _choose_burn_in(chains: np.ndarray, rhat_thresh: float = 1.05) -> int:
    """
    Choose a global burn-in index combining split-Rhat over time across chains
    and per-chain Geweke diagnostics. We take the max of:
      - earliest t where Rhat(t) <= rhat_thresh and stays under for a persistence window
      - median of per-chain Geweke burn-ins
    """
    C, T = chains.shape
    # Rhat-based
    r = _rhat_trace(chains, start=max(20, T // 50), step=max(1, T // 500))
    start_t = max(20, T // 50)
    # persistence: at least 5% of chain or 50 samples, whichever is smaller but >= 10
    persist = max(10, min(50, T // 20))
    burn_rhat = T // 2
    for idx, val in enumerate(r):
        t = start_t + idx * max(1, T // 500)
        if np.isfinite(val) and val <= rhat_thresh:
            # Check persistence
            tail = r[idx : idx + persist // max(1, T // 500)]
            if tail.size > 0 and np.all(np.isfinite(tail)) and np.all(tail <= rhat_thresh):
                burn_rhat = t
                break

    # Geweke per chain
    burns = np.array([_geweke_burn_in(chains[c]) for c in range(C)])
    burn_geweke = int(np.median(burns))

    return int(min(T - 5, max(burn_rhat, burn_geweke)))


def analyze_chains(
    E: np.ndarray,
    step: float = 1.0,
    rhat_threshold: float = 1.05,
    return_traces: bool = False,
) -> Dict[str, Any]:
    """
    Analyze multiple chains of an observable (e.g., energy) sampled over iterations.

    Parameters
    ----------
    E : np.ndarray
        Array of shape (n_chains, n_time) with per-iteration values.
    step : float
        Iteration step size (e.g., 1 if every iteration recorded; or MC sweeps per sample).
    rhat_threshold : float
        Threshold for split-Rhat to declare convergence (default 1.05).
    return_traces : bool
        If True, include R-hat trace and per-chain ACF/IAT traces (may be large).

    Returns
    -------
    stats : dict with keys
        - n_chains, n_time
        - burn_in_index, burn_in_time
        - rhat_now (R-hat at full length)
        - tau_per_chain : np.ndarray [C]
        - tau_mean
        - ess_per_chain : np.ndarray [C]
        - ess_total
        - chain_means_post, chain_vars_post
        - rhat_trace (optional)
    """
    E = np.asarray(E, dtype=float)
    assert E.ndim == 2, "E must have shape (n_chains, n_time)"
    C, T = E.shape

    # Burn-in
    burn = _choose_burn_in(E, rhat_thresh=rhat_threshold)
    post = E[:, 20:]

    # Per-chain IAT and ESS (post-burn)
    tau = np.array([_integrated_time_emcee(post[c]) for c in range(C)])
    # Guard: tau at least 1
    tau = np.clip(tau, 1.0, None)
    n_post = post.shape[1]
    ess = n_post / tau
    ess_total = float(np.sum(ess))

    # R-hat at full length
    rhat_full = _split_rhat(E)

    stats: Dict[str, Any] = dict(
        n_chains=C,
        n_time=T,
        burn_in_index=int(burn),
        burn_in_time=float(burn * step),
        rhat_now=float(rhat_full) if rhat_full is not None else np.nan,
        tau_per_chain=tau,
        tau_mean=float(np.mean(tau)),
        ess_per_chain=ess,
        ess_total=ess_total,
        chain_means_post=np.mean(post, axis=1),
        chain_vars_post=np.var(post, axis=1, ddof=1) if n_post > 1 else np.zeros(C),
    )
    if return_traces:
        stats["rhat_trace"] = _rhat_trace(E, start=max(20, T // 50), step=max(1, T // 500))

    return stats


def get_autocorrelation_states(vstate, H, sweep_size_divsor = 6, length = 64, n_chains=6):
    # Init new sampler 
    sampler_old = vstate.sampler
    sweep_size = sampler_old.sweep_size // sweep_size_divsor
    sampler = sampler_old.replace(sweep_size=sweep_size, n_chains=n_chains)

    model = vstate.model
    params = vstate.variables
    state = vstate.sampler_state
    state = state.replace(𝜎= state.𝜎[:n_chains, ...], 
                          n_accepted_proc=state.n_accepted_proc[:n_chains, ...], 
                          log_prob = state.log_prob[:n_chains, ...])
    
    state = sampler.reset(model, params, state)

    sigma, state = sampler._sample_chain(model, params, state, length)
    sigma = sigma.reshape((-1, sigma.shape[-1]))

    get_kernel_args = lambda sig : TurboTurboOperator._get_sigma_args(sig, H)
    kernel_args = [get_kernel_args(sig[None, ...]) for sig in sigma]
    
    # Repackage kernel args 
    eta = jnp.vstack([args[1][0][None, ...] for args in kernel_args])
    mels = jnp.vstack([args[1][1][None, ...] for args in kernel_args])
    idx = jnp.stack([args[1][2][0]for args in kernel_args])
    centers = jnp.vstack([args[1][2][1] for args in kernel_args])

    kernel_args = (eta, mels, (idx, centers))

    kernel = TurboTurboOperator._get_kernel(vstate, H)

    @partial(jax.jit, static_argnames=("model"))
    def compute_expectations(model, params, sigma, kernel_args):
        # We iterate along axis 0. You used sigma[:, None, ...], so keep that shape.
        sigma_seq = sigma[:, None, ...]

        def step(carry, inputs):
            sigma_i, kernel_args_i = inputs          # sigma_i has shape (1, ...)
            val = kernel(model, params, sigma_i, kernel_args_i)
            return carry, val                        # unchanged carry, emit value

        carry0 = None
        _, expectation_vals = jax.lax.scan(step, carry0, (sigma_seq, kernel_args))
        return expectation_vals
        
    # Run the kernel for each sigma
 
    expectation_vals = compute_expectations(model, params, sigma, kernel_args)

    #expectation_vals = jnp.array([run_kernel(sig[None, ...]) for sig in sigma])
    expectation_vals = expectation_vals.reshape((n_chains, -1,))

    stats = analyze_chains(expectation_vals, step=1.0, rhat_threshold=1.05, return_traces=False)
    return expectation_vals, stats

def QGTJacobianPyTree_no_shard(
    vstate, *, mode=None, holomorphic=None,
    diag_shift=0.0, diag_scale=None, chunk_size=None,
    axis_0_is_sharded=False,   # <-- your exposed switch
):
    from netket.vqs import FullSumState

    if isinstance(vstate, FullSumState):
        samples = split_array_mpi(vstate._all_states)
        pdf = split_array_mpi(vstate.probability_distribution())
    else:
        samples, pdf = vstate.samples, None

    if samples.ndim >= 3:
        samples = jax.jit(jax.lax.collapse, static_argnums=(1, 2))(samples, 0, 2)

    if mode is None:
        mode = nkjax.jacobian_default_mode(
            vstate._apply_fun, vstate.parameters, vstate.model_state, samples,
            holomorphic=holomorphic
        )
    jac_mode = "complex" if mode == "imag" else mode
    if chunk_size is None:
        chunk_size = getattr(vstate, "chunk_size", None)

    O = nkjax.jacobian(
        vstate._apply_fun, vstate.parameters, samples, vstate.model_state,
        mode=jac_mode, pdf=pdf, chunk_size=chunk_size, dense=False,
        center=True, _sqrt_rescale=True,
        _axis_0_is_sharded=axis_0_is_sharded,   # <-- the important bit
    )

    shift, offset = to_shift_offset(diag_shift, diag_scale)
    if offset is not None:
        ndims = 1 if (mode not in ("complex", "imag")) else 2
        O, scale = rescale(O, offset, ndims=ndims)
    else:
        scale = None

    pars_struct = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), vstate.parameters
    )

    return QGTJacobianPyTreeT(
        O=O, scale=scale, mode=mode, _params_structure=pars_struct, diag_shift=shift
    )

# use it in SR
# sr = nk.optimizer.SR(qgt=qgt, ...)  # as usual


class System:
    # Remove model from permanent storage
    optimizer: Optional[object] = None
    sr_optimizer: Optional[object] = None
    
    # Store optimizer parameters for reuse
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    
    optimizer_mode : str = "SR"

    def __init__(self, hi, H, sampler, n_spins, hamiltonian_params,standard_sampler = None, width: int = 0, token_size: int = 1):
        """
        Initialize a System object
        
        Args:
            hi: Hilbert space
            H: Hamiltonian operator
            sampler: MC sampler
            n_spins: Number of spins
            hamiltonian_params: Parameters for the Hamiltonian
            width: Width parameter for custom operators (default: 0)
            token_size: Token size parameter for custom operators (default: 1)
        """
        self.hi = hi
        self.H = H
        self.sampler = sampler
        self.standard_sampler = standard_sampler

        # Vstate params 
        self.n_samples = None    
        self.n_discard_per_chain = None
        self.chunk_size = None

        self.training_history = {}
    
        # Model params
        self.n_spins = n_spins
        self.hamiltonian_params = hamiltonian_params
        
        # Custom operator params
        self.width = width
        self.token_size = token_size

    def setup_optimizer(self, 
                       init_value: float = 0.1, 
                       peak_value: float = 1.0, 
                       warmup_steps: int = 50, 
                       decay_rate: float = 0.995, 
                       diag_shift_init: float = 1e-2, 
                       diag_shift_final: float = 1e-4,
                       max_iters: int = 200, 
                       end_value : float = 0.0,
                       gradient_clip : int = None, 
                       mode :str = "SR", 
                       sharded : bool = False) -> None:
        """
        Set up the optimizer and training schedule
        
        Args:
            init_value: Initial learning rate
            peak_value: Maximum learning rate
            warmup_steps: Number of steps to warm up to peak_value
            decay_rate: Decay rate for learning rate after warm-up
            diag_shift_init: Initial diagonal shift for SR
            diag_shift_final: Final diagonal shift for SR
            max_iters: Maximum number of iterations
            mode: Optimizer mode ("SR" or "minSR")
        """
        # Store optimizer parameters for future reference
        self.optimizer_params = {
            "init_value": init_value,
            "peak_value": peak_value,
            "warmup_steps": warmup_steps,
            "decay_rate": decay_rate,
            "diag_shift_init": diag_shift_init,
            "diag_shift_final": diag_shift_final,
            "end_value" : end_value,
            "max_iters": max_iters
        }
            
        # Set up learning rate schedule
        lr_schedule = optax.warmup_exponential_decay_schedule(
            init_value=init_value,
            peak_value=peak_value,
            warmup_steps=warmup_steps,
            transition_steps=1,
            decay_rate=decay_rate,
            end_value=end_value,
        )
        
        # Set up optimizer
        optimizer = nk.optimizer.Sgd(learning_rate=lr_schedule)
        if gradient_clip is not None:
            # Add gradient clipping if specified
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(gradient_clip),
                optimizer
            )
        else: 
            self.optimizer = optimizer 
        
        # Set up diagonal shift schedule for SR
        self.ds_schedule = optax.linear_schedule(diag_shift_init, diag_shift_final, max_iters)

        if sharded: 
            qgt = partial(QGTJacobianPyTree_no_shard, mode="real", axis_0_is_sharded=True)
            qgt = partial(nk.optimizer.qgt.QGTOnTheFly, chunk_size=None)
            print("set QGT None")

            self.sr_optimizer = nk.optimizer.SR(qgt=qgt, diag_shift=self.ds_schedule)
            self.sr_optimizer = None 
        else: 
            self.sr_optimizer = nk.optimizer.SR(diag_shift=self.ds_schedule)

        if mode not in {"SR", "minSR"}:
            raise ValueError(f"Unknown optimizer mode: {mode}. Use 'SR' or 'minSR'.")
        self.optimizer_mode = mode
        

    def setup_sampling(self, n_samples: int = 512, n_discard_per_chain: int = 5, chunk_size: Optional[int] = None):
        self.n_samples = n_samples
        self.n_discard_per_chain = n_discard_per_chain
        self.chunk_size = chunk_size


    def get_variational_state(self, model, params = None, chunk_size: Optional[int] = None, seed : int = 42, sampler_seed : int = 42) -> nk.vqs.MCState:
        """
        Set up a variational state with the provided model
        
        Args:
            model: The model to use in the variational state
            n_samples: Number of samples for MC sampling
            n_discard_per_chain: Number of samples to discard per chain
            chunk_size: Chunk size for batching
            
        Returns:
            vstate: The variational state
        """
        if model is None:
            raise ValueError("Model must be provided to setup variational state")
            
        if chunk_size is None:
            chunk_size = self.chunk_size

        vstate = nk.vqs.MCState(
            self.sampler,
            model,
            n_samples=self.n_samples,
            n_discard_per_chain=self.n_discard_per_chain,
            chunk_size=chunk_size,
            variables=params,
            seed=seed,
            sampler_seed=sampler_seed
        )
        
        return vstate

    def get_driver(self, vstate, operator: Optional[nk.operator.LocalOperator] = None, minSR = False, mode = "") -> nk.driver.VMC:
        """
        Set up the VMC driver
        
        Args:
            vstate: The variational state to use
            
        Returns:
            driver: The VMC driver
        """
        if self.optimizer is None:
            raise ValueError("Optimizer must be set before setting up driver")
            
        if operator is None:
            operator = self.H
        
        if mode == "": 
            mode = "minSR" if minSR else "SR"

        if self.optimizer_mode == "minSR": 
            driver = nkx.driver.VMC_SRt(
                operator,
                self.optimizer,
                diag_shift=self.ds_schedule,
                variational_state=vstate,
            )
        elif self.optimizer_mode == "SR": 
            driver = nk.driver.VMC(
                operator,
                self.optimizer,
                variational_state=vstate,
                preconditioner=self.sr_optimizer,
            )
        else: 
            raise ValueError(f"Unknown driver mode: {mode}. Use 'minSR' or 'SR'.")

        return driver

    def setup_observables(self, use_custom_operators: bool = False) -> Dict[str, nk.operator.LocalOperator]:
        """
        Set up common observables for the system
        
        Args:
            use_custom_operators: Whether to use custom (faster) operators
        
        Returns:
            observables: Dictionary of observables
        """
        N = self.hi.size
        
        if use_custom_operators:
            # Use custom operators for better performance
            observables = {
                "staggered_magnetization": staggered_magnetization_op(
                    self.hi, self.width, token_size=self.token_size
                ),
                "magnetization": magnetization_op(
                    self.hi, self.width, token_size=self.token_size
                ),
                "magnetization_squared": magnetization_sqr_op(
                    self.hi, self.width, token_size=self.token_size
                ),
                "staggered_magnetization_squared": staggered_magnetization_sqr_op(
                    self.hi, self.width, token_size=self.token_size
                )
            }
        else:
            # Use standard NetKet operators
            # Renyi entropy
            #renyi = nkx.observable.Renyi2EntanglementEntropy(
            #    self.hi, np.arange(0, N // 2 + 1, dtype=int)
            #)
            
            # Magnetization
            mags = sum([(-1) ** i * sigmaz(self.hi, i) / N for i in range(N)])
            magnet = sum([sigmaz(self.hi, i) / N for i in range(N)])
            
            observables = {
                "staggered_magnetization": mags,
                "magnetization": magnet,
                "magnetization_squared": magnet @ magnet,
                "staggered_magnetization_squared": mags @ mags
            }
        
        return observables
    
    def train(self, 
              model,
              run_id: Optional[str] = None,
              n_iter: int = 200, 
              callback: Optional[List[Callable]] = None, 
              params: Optional[Dict[str, Any]] = None,
              chunk_size: Optional[int] = None,
              operator: Optional[nk.operator.LocalOperator] = None,
              n_samples: Optional[int] = None,
              show_progress: bool = True, 
              skip_observables: bool = False,
              minSR : bool = False,
              n_discard_per_chain : int = 5, 
              mode : str = "",
              use_custom_operators: bool = False, 
              gather_autocorrelation_stats: bool = True,
              seed : int = 42, 
              sampler_seed : int = 42,
              compute_correlators: bool = False,
              compute_structure_factor: Optional[bool] = None) -> Tuple[nk.logging.RuntimeLog, Any]:
        """
        Run training with the given model and return the log and best keeper
        
        Args:
            model: The model to train
            run_id: Optional identifier for this training run
            n_samples: Number of samples for MC sampling
            n_iter: Number of iterations
            callback: List of callback functions
            show_progress: Whether to show progress bar
            skip_observables: Skip all observable measurements (including correlators/structure factor)
            compute_correlators: Whether to compute <sigma^z_j sigma_{j+r}^z> and structure factor metrics
            compute_structure_factor: Whether to compute the structure factor explicitly (defaults to compute_correlators)
            use_custom_operators: Whether to use custom (faster) operators for metrics
            
        Returns:
            log: The runtime log
            keeper: The best iteration keeper
        """
        if self.optimizer is None:
            # Use default optimizer settings if not set
            self.setup_optimizer()

        if self.n_samples is None or n_samples is not None:
            self.setup_sampling(n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

        # Set up variational state
        vstate = self.get_variational_state(model, chunk_size=chunk_size, params=params, seed=seed, sampler_seed=sampler_seed)

        sampler = vstate.sampler
        schedule_callback = None
        if hasattr(sampler, "has_dynamic_sweep_schedule") and sampler.has_dynamic_sweep_schedule:
            new_sampler_state = sampler.prepare_initial_sweep_state(vstate.sampler_state)
            if new_sampler_state is not vstate.sampler_state:
                assign_fn = getattr(sampler, "_assign_sampler_state", None)
                if assign_fn is not None:
                    vstate, _ = assign_fn(vstate, new_sampler_state)
                elif hasattr(vstate, "replace"):
                    vstate = vstate.replace(sampler_state=new_sampler_state)
                else:
                    try:
                        setattr(vstate, "sampler_state", new_sampler_state)
                    except AttributeError:
                        try:
                            object.__setattr__(vstate, "_sampler_state", new_sampler_state)
                        except AttributeError:
                            vstate.__dict__["sampler_state"] = new_sampler_state
            schedule_callback = sampler.make_sweep_schedule_callback()
        
        # Set up driver
        driver = self.get_driver(vstate, operator=operator, minSR=minSR, mode=mode)

        # Set up logging
        log = nk.logging.RuntimeLog()
        
        # Set up best iteration keeper
        keeper = BestIterKeeper(self.H, self.hi.size, 1e-8)
        
        # Combine callbacks
        all_callbacks = []
        if schedule_callback is not None:
            all_callbacks.append(schedule_callback)
        all_callbacks.append(keeper.update)
        if callback is not None:
            all_callbacks.extend(callback)
            
        # Run the optimization
        start_time = time.time()
        driver.run(n_iter=n_iter, out=log, callback=all_callbacks, show_progress=show_progress)
        end_time = time.time()
        vstate = driver.state
        
        training_time = end_time - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Generate a unique run ID if not provided
        if run_id is None:
            run_id = f"run_{len(self.training_history) + 1}"
        
        # Compute metrics
        metrics = self.compute_metrics(
            keeper,
            use_custom_operators=use_custom_operators,
            skip_observables=skip_observables,
            compute_correlators=compute_correlators,
            compute_structure_factor=compute_structure_factor,
            model=model,
        )

        # Store in training history
        run_results = {
            "run_id": run_id,
            "log": log,
            "keeper": keeper,
            "metrics": metrics,
            "energy_trajectory": log.data["Energy"].Mean,
            "variance_trajectory": log.data["Energy"].Variance,
            "acceptance_rate": log["acceptance"].value,
            "training_time": training_time,
            "n_samples": self.n_samples,
            "n_iter": n_iter, 
            "n_chains": vstate.sampler.n_chains, 
        }

        if gather_autocorrelation_stats:
            energy, stats = get_autocorrelation_states(vstate, self.H)
            run_results["autocorrelation_energy"] = energy
            run_results["autocorrelation_stats"] = stats
        
        self.training_history[run_id] = run_results
        
        return log, keeper
    


    def save_training_history(self, 
                          folder_path: str = "./training_results",
                          run_id: Optional[str] = None,
                          save_model_params: bool = False,
                          save_mode: str = "last") -> None:
        """
        Save training history to disk

        Args:
            folder_path: Folder to save results in
            run_id: Specific run ID to save, if None uses save_mode
            save_model_params: Whether to save model parameters
            save_mode: Mode for selecting runs to save if run_id is None:
                    'last' - save only the last run
                    'all' - save all runs
        """
        import os, json, pickle
        from datetime import datetime
        import numpy as np

        def make_serializable(obj):
            # If the object has a 'tolist' method (e.g. numpy or JAX arrays), use it
            if hasattr(obj, "tolist"):
                try:
                    return obj.tolist()
                except Exception:
                    pass
            # Numbers (including np/jnp scalars)
            try:
                if isinstance(obj, (np.generic, jnp.generic)):
                    return obj.item()
            except Exception:
                pass
            if isinstance(obj, (float, int, bool)):
                return obj
            # Dtypes and scalar types
            if isinstance(obj, (np.dtype, jnp.dtype)):
                return obj.name
            if isinstance(obj, type):
                # Handle numpy/jax scalar type classes (e.g., jnp.float32) and other dtype metaclasses
                try:
                    if issubclass(obj, (np.generic, jnp.generic)):
                        return str(obj)
                except Exception:
                    pass
                module = getattr(obj, "__module__", "")
                if module.startswith("jax"):
                    return str(obj)
                name = getattr(obj, "__name__", None)
                return name if name is not None else str(obj)
            # Recursively process dictionaries
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            # Process lists and tuples recursively
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(make_serializable(item) for item in obj)
            # Fallback: try to convert via np.asarray
            try:
                arr_obj = np.asarray(obj)
                if arr_obj.shape == () and not isinstance(obj, str):
                    return arr_obj.item()
            except Exception:
                pass
            # Otherwise, return the object as is
            return obj

        # Create folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        
        # Timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine which runs to save
        runs_to_save = {}
        if run_id is not None:
            if run_id in self.training_history:
                runs_to_save[run_id] = self.training_history[run_id]
            else:
                raise ValueError(f"Run ID '{run_id}' not found in training history")
        elif save_mode == "last" and self.training_history:
            last_run_id = list(self.training_history.keys())[-1]
            runs_to_save[last_run_id] = self.training_history[last_run_id]
        elif save_mode == "all":
            runs_to_save = self.training_history
        else:
            raise ValueError("No runs to save or invalid save_mode. Use 'last' or 'all'.")
        
        for rid, run_data in runs_to_save.items():
            # Create a serializable copy of run_data by recursively converting arrays
            serializable_data = {
                k: make_serializable(v) 
                for k, v in run_data.items() 
                if k not in ["log", "keeper"]
            }
            
            # Add system info
            serializable_data["system_info"] = {
                "n_spins": float(self.n_spins),
                "hamiltonian_params": self.hamiltonian_params,
                "optimizer_params": self.optimizer_params
            }
            
# Get model configuration if available
            if "keeper" in run_data:
                try:
                    best_state = run_data["keeper"].best_state
                    
                    # Extract model config if the model has a get_config method
                    if hasattr(best_state.model, "get_config"):
                        model_config = best_state.model.get_config()
                        serializable_data["model_config"] = make_serializable(model_config)
                        print(f"Added model configuration to saved data for run '{rid}'")
                except Exception as e:
                    print(f"Failed to extract model configuration: {str(e)}")
            
            # Save the run data
            file_path = os.path.join(folder_path, f"{rid}_{timestamp}.json")
            with open(file_path, "w") as f:
                json.dump(serializable_data, f, indent=2)
            
            print(f"Saved run '{rid}' to {file_path}")
            
            # Optionally save model parameters
            if save_model_params and "keeper" in run_data:
                try:
                    best_state = run_data["keeper"].best_state

                    # Save the raw parameters
                    params_path = os.path.join(folder_path, f"{rid}_{timestamp}_params.pkl")
                    with open(params_path, 'wb') as f:
                        pickle.dump(best_state.parameters, f)
                    print(f"Saved model parameters for run '{rid}' to {params_path}")
                except Exception as e:
                    print(f"Failed to save model parameters: {str(e)}")

    def compute_metrics(
        self,
        keeper: Any,
        use_custom_operators: bool = False,
        skip_observables: bool = False,
        compute_correlators: bool = False,
        compute_structure_factor: Optional[bool] = None,
        model=None,
    ) -> Dict[str, Any]:
        """
        Compute metrics using the best state from the keeper
        
        Args:
            keeper: The best iteration keeper
            use_custom_operators: Whether to use custom (faster) operators
            compute_correlators: Whether to compute <sigma_j^z sigma_{j+r}^z> for r=1..N/2
            compute_structure_factor: Whether to compute the ZZ structure factor. If None,
                uses the value of compute_correlators.
            
        Returns:
            metrics: Dictionary of metrics
        """
        if compute_structure_factor is None:
            compute_structure_factor = compute_correlators
        
        # Compute metrics
        metrics = {}
        metrics["best_energy"] = keeper.best_energy
        metrics["vscore"] = keeper.vscore

        vstate = keeper.best_state
        
        if self.standard_sampler is not None: 
            assert model is not None, "Model must be provided for standard sampler"
            params = vstate.variables 
            vstate = self.get_variational_state(model, params = params)

            metrics["best_energy"] = vstate.expect(self.H).mean
            
        # Set up observables
        if not skip_observables:
            observables = self.setup_observables(use_custom_operators=use_custom_operators)

            # Compute observable expectations
            for name, obs in observables.items():
                metrics[name] = np.real(vstate.expect(obs).mean)

            if compute_correlators or compute_structure_factor:
                # Flatten samples to (num_samples, n_spins)
                samples = np.asarray(vstate.samples)
                if samples.ndim == 3:
                    samples = samples.reshape(-1, samples.shape[-1])
                elif samples.ndim == 2:
                    pass
                elif samples.ndim == 1:
                    samples = samples[None, :]
                else:
                    raise ValueError(f"Unsupported sample shape for correlators: {samples.shape}")

                samples_jnp = jnp.asarray(samples, dtype=jnp.float32)
                if compute_correlators:
                    corr_vals = np.array(
                        zz_correlators(samples_jnp, max_r=self.hi.size // 2)
                    )
                    metrics["zz_correlator"] = np.real(np.mean(corr_vals, axis=0))

                if compute_structure_factor:
                    sf_vals = np.array(structure_factor_zz(samples_jnp))
                    metrics["structure_factor"] = np.real(np.mean(sf_vals, axis=0))
        
        return metrics
    
    def compute_exact_diagonalization(self, order: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the exact diagonalization of the Hamiltonian if possible
        
        Args:
            order: Number of eigenstates to compute
            
        Returns:
            eigenvalues: The eigenvalues
            eigenvectors: The eigenvectors
        """
        if self.hi.size > 20:
            raise ValueError("Exact diagonalization only available for systems with up to 20 spins")
        
        return get_eigvals(Hamiltonian=self.H, order=order, eigenvecs=True)
    
    
    def compare_models(self, 
                       models: Dict[str, Any], 
                       training_config: Dict[str, Any] = None,
                       use_custom_operators: bool = False) -> Dict[str, Any]:
        """
        Train and compare multiple models on this system
        
        Args:
            models: Dictionary of models to compare (name -> model)
            training_config: Configuration for training
            use_custom_operators: Whether to use custom (faster) operators for metrics
            
        Returns:
            comparison_results: Dictionary with results for all models
        """
        if training_config is None:
            training_config = {
                "n_samples": 512,
                "n_iter": 200,
                "lr_peak": 1.0,
                "warmup_steps": 50
            }
        
        # Results dictionary
        comparison_results = {}
        
        # Check if exact diagonalization is possible
        compute_exact = self.hi.size <= 20
        exact_results = None
        
        for name, model in models.items():
            print(f"\nTraining model: {name}")
            print("-" * 50)
            
            # Run training with this model
            log, keeper = self.train(
                model=model,
                run_id=name,
                n_samples=training_config.get("n_samples", 512),
                n_iter=training_config.get("n_iter", 200),
                show_progress=True,
                use_custom_operators=use_custom_operators
            )
            
            # Store results in comparison dictionary
            comparison_results[name] = {
                "log": log,
                "keeper": keeper,
                "metrics": self.compute_metrics(keeper, use_custom_operators=use_custom_operators)
            }
            
            # Store exact results (only need to do this once)
            if compute_exact and exact_results is None:
                try:
                    exact_eigenvalues, exact_eigenvectors = self.compute_exact_diagonalization()
                    exact_results = {
                        "exact_ground_energy": exact_eigenvalues[0],
                        "exact_eigenvalues": exact_eigenvalues,
                        "exact_eigenvectors": exact_eigenvectors
                    }
                except Exception as e:
                    print(f"Warning: Could not compute exact diagonalization: {e}")
                    exact_results = None
        # Add comparison metadata
        comparison_results["comparison_id"] = f"comparison_{len([k for k in self.training_history.keys() if k.startswith('comparison_')]) + 1}"
        comparison_results["models_compared"] = list(models.keys())
        comparison_results["training_config"] = training_config
        if exact_results:
            comparison_results["exact_results"] = exact_results
        
        # Store in training history
        self.training_history[comparison_results["comparison_id"]] = comparison_results
        
        return comparison_results

    def get_available_operators(self) -> Dict[str, str]:
        """
        Get information about available operator types
        
        Returns:
            Dictionary with operator types and descriptions
        """
        return {
            "standard": "Standard NetKet operators (default)",
            "custom": f"Custom operators with width={self.width}, token_size={self.token_size} (faster)"
        }
    
    def benchmark_operators(self, model, n_samples: int = 100) -> Dict[str, float]:
        """
        Benchmark the performance difference between standard and custom operators
        
        Args:
            model: The model to benchmark with
            n_samples: Number of samples for the benchmark
            
        Returns:
            Dictionary with timing results
        """
        import time
        
        # Set up a small variational state for benchmarking
        vstate = self.get_variational_state(model, chunk_size=None)
        vstate.n_samples = n_samples
        
        # Benchmark standard operators
        start_time = time.time()
        observables_standard = self.setup_observables(use_custom_operators=False)
        for name, obs in observables_standard.items():
            _ = vstate.expect(obs)
        standard_time = time.time() - start_time
        
        # Benchmark custom operators
        start_time = time.time()
        observables_custom = self.setup_observables(use_custom_operators=True)
        for name, obs in observables_custom.items():
            _ = vstate.expect(obs)
        custom_time = time.time() - start_time
        
        return {
            "standard_time": standard_time,
            "custom_time": custom_time,
            "speedup_factor": standard_time / custom_time if custom_time > 0 else float('inf'),
            "n_samples": n_samples
        }

def setup_system_long_range_ising(
    N=20,
    J=1.0,
    alpha=5.0,
    width=5,
    model=None,
    chunk_size=None,
    hz=0,
    fast=False,
    fast_sampler=False,
    simple_typewriter=False,
    lut_default=0.1,
    dyn_tol_enabled=True,
    use_reduced_precision: bool = False,
    token_size: int = 1,
    accept_max=jnp.inf,
    interval_mult=1,
    override_standard_sampler=False,
    n_chains=None,
    sweep_size=None,
    accept_tol=0.01,
    lut_multiply: float = 1.5,
    sweep_ramp: cs.SweepRampConfig | Callable[[int], int] | dict | None = None,
) -> System:
    """
    Set up a long-range Ising model system
    
    Args:
        N: Number of spins
        J: Coupling constant
        alpha: Power-law decay
        width: Width of the interaction
        chunk_size: Chunk size for batching
        sweep_ramp: Optional schedule configuration (dict, callable, or `SweepRampConfig`)
            controlling the sweep size during optimisation.

    Returns:
        system: A System object
    """
    # Define the Hilbert space
    hi = nk.hilbert.Spin(s=1/2, N=N)

    # Define the sampler
    rule1 = nk.sampler.rules.LocalRule()
    rule2 = InvertMagnetization()  # from https://arxiv.org/html/2407.04773v1
    pinvert = 0.25
    pflip = 1 - pinvert

    if isinstance(sweep_ramp, dict):
        sweep_ramp = cs.SweepRampConfig(**sweep_ramp)


    if n_chains is None:
        sampler = nk.sampler.MetropolisSampler(
            hi, nk.sampler.rules.MultipleRules([rule1, rule2], [pflip, pinvert]), 
            sweep_size=sweep_size,
        )
    else: 
        sampler = nk.sampler.MetropolisSampler(
            hi, nk.sampler.rules.MultipleRules([rule1, rule2], [pflip, pinvert]),
            n_chains=n_chains, 
            sweep_size=sweep_size,
        )


    standard_sampler = None
    if fast_sampler:
        assert model is not None, "Model must be provided for fast sampler"
        standard_sampler = sampler

        model_sampler = force_link_tensor_true(model)
        sampler = cs.MetropolisSampler(hi, nk.sampler.rules.MultipleRules([rule1, rule2], [pflip, pinvert]),
                                    global_proposal=True, 
                                    model=model_sampler, 
                                    width=width,
                                    sweep_size=sweep_size,                  
                                    token_size=token_size, 
                                    n_chains=n_chains,
                                    accept_tol=accept_tol, 
                                    output_size=model.logcosh_hidden, 
                                    simple_typewriter = simple_typewriter, 
                                    interval_mult=interval_mult, 
                                    max_accept=accept_max, 
                                    lut_default=lut_default, 
                                    lut_floor=5e-3, 
                                    lut_multiply=lut_multiply,
                                    dyn_tol_enabled = dyn_tol_enabled,
                                    use_reduced_precision=use_reduced_precision,
                                    sweep_ramp=sweep_ramp,
                                )

        if override_standard_sampler:
            standard_sampler = sampler
    if fast:
        H = hamiltonian_tfim(hi, J, alpha, width, token_size=token_size, hz=hz)
    else: 
        H = get_Hamiltonian(N=N, J=J, alpha=alpha, sym_field=hz)

    if sweep_size is None:
        sweep_size = N 

    # Create the system
    system = System(
        hi=hi, 
        H=H, 
        sampler=sampler, 
        standard_sampler = standard_sampler, 
        n_spins=N, 
        hamiltonian_params={"J": J, "alpha": alpha},
        width=width,
        token_size=token_size, 
    )
    
    # Set up default optimizer
    system.setup_optimizer()
    
    return system


def setup_system_j1j2(
    N: int = 20,
    J1: float = 0.0,
    J2: float = 0.0,
    width: int = 1,
    model=None,
    chunk_size=None,
    hz: float = 0.0,
    fast : bool = False, 
    fast_sampler: bool = False,
    simple_typewriter: bool = False,
    lut_default: float = 0.1,
    dyn_tol_enabled: bool = True,
    use_reduced_precision: bool = False,
    token_size: int = 1,
    accept_max: float = jnp.inf,
    interval_mult: int = 1,
    override_standard_sampler: bool = False,
    n_chains: int | None = None,
    sweep_size: int | None = None,
    accept_tol: float = 0.01,
    lut_multiply: float = 1.5,
    sweep_ramp: cs.SweepRampConfig | Callable[[int], int] | dict | None = None,
) -> System:
    """
    Convenience setup that mirrors `setup_system_long_range_ising` but wires in
    the J1–J2 XXZ Hamiltonian implemented in `dysonnet.custom_operator`.
    Only the fast operator mode is supported, though the sampler can be either
    the vanilla Metropolis sampler or the custom fast sampler.
    """
    del chunk_size  # Unused, kept for API symmetry.


    # Define custom graph
    edge_colors = []
    for i in range(N):
        edge_colors.append([i, (i + 1) % N, 1])
        edge_colors.append([i, (i + 2) % N, 2])

    # Define the netket graph object
    g = nk.graph.Graph(edges=edge_colors)

    # Spin based Hilbert Space
    hi = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=g.n_nodes)


    if isinstance(sweep_ramp, dict):
        sweep_ramp = cs.SweepRampConfig(**sweep_ramp)

    if n_chains is None:
        sampler = nk.sampler.MetropolisExchange(hilbert=hi, graph=g, d_max=2)
    else:
        sampler = nk.sampler.MetropolisExchange(hilbert=hi, 
            graph=g, 
            d_max=2, 
            n_chains=n_chains,
            sweep_size=sweep_size,
        )
    standard_sampler = None

    if fast_sampler:
        raise NotImplementedError("Fast sampler for J1-J2 model is not implemented yet.")
        assert model is not None, "Model must be provided for fast sampler"
        standard_sampler = sampler

        model_sampler = force_link_tensor_true(model)
        sampler = cs.MetropolisSampler(
            hi,
            nk.sampler.rules.MultipleRules([rule1, rule2], [pflip, pinvert]),
            global_proposal=True,
            model=model_sampler,
            width=width,
            sweep_size=sweep_size,
            token_size=token_size,
            n_chains=n_chains,
            accept_tol=accept_tol,
            output_size=model.logcosh_hidden,
            simple_typewriter=simple_typewriter,
            interval_mult=interval_mult,
            max_accept=accept_max,
            lut_default=lut_default,
            lut_floor=5e-3,
            lut_multiply=lut_multiply,
            dyn_tol_enabled=dyn_tol_enabled,
            use_reduced_precision=use_reduced_precision,
            sweep_ramp=sweep_ramp,
        )

        if override_standard_sampler:
            standard_sampler = sampler


    if fast:
        H = hamiltonian_j1j2(
            hi,
            J1_xy=-J1,
            J1_z=J1,
            J2_xy=J2,
            J2_z=J2,
            width=width,
            hz=hz,
            token_size=token_size,
            z2_project=False,
        )
    else:
        # 2. Define Operators
        sigmaz = np.array([[1, 0], [0, -1]])
        mszsz = np.kron(sigmaz, sigmaz)
        
        # Exchange matrix from snippet
        exchange = np.asarray([[0, 0, 0, 0],
                            [0, 0, 2, 0],
                            [0, 2, 0, 0],
                            [0, 0, 0, 0]])

        # Using the signs from the LAST provided function in the prompt:
        # Color 1: J1*SzSz, -J1*Exchange
        # Color 2: J2*SzSz,  J2*Exchange
        bond_operator = [
            (J1 * mszsz).tolist(),      
            (J2 * mszsz).tolist(),      
            (-J1 * exchange).tolist(),  
            (J2 * exchange).tolist(),   
        ]
        bond_colors = [1, 2, 1, 2]

        H = nk.operator.GraphOperator(
            hi, graph=g, bond_ops=bond_operator, bond_ops_colors=bond_colors
        )
        

    if sweep_size is None:
        sweep_size = N

    system = System(
        hi=hi,
        H=H,
        sampler=sampler,
        standard_sampler=standard_sampler,
        n_spins=N,
        hamiltonian_params={
            "J1": J1,
            "J2": J2,
            "hz": hz,
        },
        width=width,
        token_size=token_size,
    )
    system.setup_optimizer()

    return system


# Capture the model activations 
def capture_mixer_outputs(module: nn.Module, method_name: str) -> bool:
    """
    Capture the output of the __call__ method for DysonBlock instances.
    """
    # Check if the module is an instance of the block types we care about
    is_target_block = isinstance(module, DysonBlock)

    # Check if the method being called is the main forward pass method
    is_call_method = (method_name == '__call__')

    return is_target_block and is_call_method


def renormalize_act_norm(model, params, sigma, epsilon = 1e-6):
    """
    Renormalize the activation normalization parameters of the model.

    Args:
        model: The model instance.
        params: The model parameters.
        sigma: The activation normalization parameter to renormalize.

    Returns:
        The renormalized parameters.
    """
    # Deep copy to avoid modifying the original params
    new_params = copy.deepcopy(params)
    for block_index in range(model.n_blocks):
        _, intermediates = model.apply(new_params, sigma, mutable=["intermediates"], capture_intermediates=True)

        block_name = f"blocks_{block_index}"
        x = intermediates["intermediates"][block_name]["LayerNorm_x"]["__call__"][0]
        reduction_axis = tuple(range(x.ndim - 1))
        mean = jnp.mean(x, axis=reduction_axis, keepdims=True)
        var  = jnp.var(x, axis=reduction_axis, keepdims=True)

        scale = (1.0 / jnp.sqrt(var + epsilon)).reshape(params["params"][block_name]["LayerNorm_x"]["scale"].shape)
        bias = (-mean / jnp.sqrt(var + epsilon)).reshape(new_params["params"][block_name]["LayerNorm_x"]["bias"].shape)

        prev_scale = new_params["params"][block_name]["LayerNorm_x"]["scale"]
        prev_bias = new_params["params"][block_name]["LayerNorm_x"]["bias"]

        new_params["params"][block_name]["LayerNorm_x"]["scale"] = scale*prev_scale 
        new_params["params"][block_name]["LayerNorm_x"]["bias"] = scale*prev_bias +  bias

    return new_params

def renormalize_act_norm_vstate(model, params, vstate, epsilon = 1e-6):
    """
    Renormalize the activation normalization parameters using the samples from a variational state.

    Args:
        model: The model instance.
        params: The model parameters.
        vstate: The variational state containing samples.
        epsilon: Small constant for numerical stability.

    Returns:
        The renormalized parameters.
    """
    sigma = vstate.samples.reshape((-1, )+ vstate.samples.shape[2:])
    new_params = renormalize_act_norm(model, params, sigma, epsilon=epsilon)
    return new_params



# Usage examples for custom operators:
#
# Example 1: Setup system with custom operators enabled
# system = setup_system_long_range_ising(N=20, J=4.75, alpha=6.0, width=5, token_size=1)
#
# Example 2: Train with custom operators
# log, keeper = system.train(
#     model=your_model,
#     n_iter=200,
#     n_samples=512,
#     use_custom_operators=True  # Enable custom operators for faster computation
# )
#
# Example 3: Compare operators performance
# benchmark_results = system.benchmark_operators(your_model, n_samples=100)
# print(f"Speedup factor: {benchmark_results['speedup_factor']:.2f}x")
#
# Example 4: Compare models with custom operators
# models = {"model1": model1, "model2": model2}
# comparison_results = system.compare_models(
#     models, 
#     training_config={"n_iter": 100, "n_samples": 256},
#     use_custom_operators=True
# )
