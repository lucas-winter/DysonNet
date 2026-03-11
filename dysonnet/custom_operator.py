# file: clipped_flip_operator.py
import jax
import jax.numpy as jnp
from functools import partial
import netket as nk
from netket.operator import AbstractOperator
import math
import dysonnet.DysonNQS as mnqs
import inspect
from flax.core import unfreeze, freeze # Make sure these are imported
from typing import Any, Optional
from dysonnet.partial_evaluation import slice_array_jit, slice_tokens_jit
import functools

def shift_arrays(eta, sigma, idx, c, *, width: int, token_size: int):
    """
    Roll both the full configurations and the sliced windows while keeping the
    central spin consistent with the original flip pattern.
    """
    L = sigma.shape[-1]
    sigma_shifted = jnp.roll(sigma, 1, axis=-1)

    # Offsets inside each window (same convention as operator construction)
    window_len = eta.shape[1]
    offsets = jnp.arange(window_len, dtype=jnp.int32) - width * token_size

    # Build the full eta configurations to capture every flipped site
    row_idx = jnp.arange(eta.shape[0], dtype=idx.dtype)
    c_mod = c % L
    centers_spin = (c_mod // token_size) * token_size
    positions = (centers_spin[:, None] + offsets[None, :]) % L

    sigma_rows = sigma[idx, :]
    eta_full = sigma_rows.at[row_idx[:, None], positions].set(eta)

    # Shift the full eta configuration, then re-slice the window in the shifted frame
    eta_full_shifted = jnp.roll(eta_full, 1, axis=-1)
    c_shifted = (c_mod + 1) % L
    centers_spin_shifted = (c_shifted // token_size) * token_size

    eta_shifted = slice_tokens_jit(
        eta_full_shifted,
        row_idx,
        centers_spin_shifted,
        width=width,
        token_size=token_size,
    )

    return sigma_shifted, eta_shifted

def duplicate_model_via_factory(
    original_model: mnqs.DysonNet, # Type hint for clarity
    arg_name_to_change: str,
    new_value: Any,
    factory_create_method # Pass the factory method: mnqs.DysonNetFactory.create_model
) -> mnqs.DysonNet:
    """
    Creates a new DysonNet model instance using the DysonNetFactory,
    copying parameters from an original model and changing one.

    Args:
        original_model: The existing DysonNet instance.
        arg_name_to_change: The name of the factory argument to change.
        new_value: The new value for this argument.
        factory_create_method: The DysonNetFactory.create_model method.

    Returns:
        A new DysonNet instance.
    """
    
    # 1. Get all configurations from the original model
    model_configs = original_model.get_config()

    # 2. Prepare arguments for the factory's create_model method
    factory_args = {}
    factory_signature_params = inspect.signature(factory_create_method).parameters

    # Map from model_configs keys (or derived) to factory argument names
    # This needs to be aligned with factory_create_method's signature
    
    # Primitives that map directly or with minor name changes
    factory_args['token_size'] = model_configs.get('token_size')
    factory_args['embedding_dim'] = model_configs.get('embedding_dim')
    factory_args['n_blocks'] = model_configs.get('n_blocks')
    factory_args['hidden_dim'] = model_configs.get('hidden_dim')
    factory_args['dropout_rate'] = model_configs.get('dropout_rate')
    factory_args['use_gated_mlp'] = model_configs.get('use_gated_mlp')
    factory_args['s4_seq_length'] = model_configs.get('seq_length') # Name change
    factory_args['s4_init_type'] = model_configs.get('s4_init_type')
    factory_args['s4_decay_scale'] = model_configs.get('s4_decay_scale')
    factory_args['s4_freq_scale'] = model_configs.get('s4_freq_scale')
    factory_args['s4_long_range_frac'] = model_configs.get('s4_long_range_frac')
    factory_args['s4_l_width'] = model_configs.get('s4_l_width')
    factory_args['use_logcosh_output'] = model_configs.get('use_logcosh_output')
    factory_args['logcosh_hidden'] = model_configs.get('logcosh_hidden')
    factory_args['use_complex'] = model_configs.get('use_complex') # For LogCosh & S4 default complex
    factory_args['complex_output'] = model_configs.get('complex_output')
    factory_args['bidirectional'] = model_configs.get('bidirectional')
    factory_args['use_convolution'] = model_configs.get('use_convolution')
    factory_args['s4_include_interblock'] = model_configs.get('s4_include_interblock')
    factory_args['partial_evaluation'] = model_configs.get('partial_evaluation')

    # Arguments derived from the model's internal full configs
    s4_config_full = model_configs.get('s4_config_full', {})

    factory_args['conv_kernel_size'] = s4_config_full.get('conv_kernel_size', factory_signature_params['conv_kernel_size'].default)
    factory_args['conv_layer_number'] = s4_config_full.get('conv_layer_number', factory_signature_params['conv_layer_number'].default)
    factory_args['s4_states'] = s4_config_full.get('d_state', factory_signature_params['s4_states'].default)
    factory_args['s4_use_gating'] = s4_config_full.get('use_gating', factory_signature_params['s4_use_gating'].default)

    # For the factory's s4_config argument:
    # These are for *additional* overrides not covered by explicit factory args.
    # The factory defines its own defaults and then applies these.
    # We take the full configs from the original model and remove keys that
    # are now being passed as explicit primitive arguments to the factory.
    s4_primitives_for_factory = {
        'd_state', 'l_max', 'l_width', 'use_conv', 'conv_kernel_size', 'conv_layer_number',
        'use_gating', 'bidirectional', 'complex_params', 'dropout_rate', 
        'init_type', 'decay_scale', 'freq_scale', 'long_range_frac'
    }
    s4_override_for_factory = {
        k: v for k, v in s4_config_full.items() if k not in s4_primitives_for_factory
    }
    if s4_override_for_factory:
        factory_args['s4_config'] = freeze(s4_override_for_factory)


    # Ensure all required factory args are present, using defaults from factory if not mapped
    final_call_args = {}
    for p_name, p_obj in factory_signature_params.items():
        if p_name in factory_args and factory_args[p_name] is not None: # Use if we mapped it
            final_call_args[p_name] = factory_args[p_name]
        elif p_obj.default is not inspect.Parameter.empty: # Use factory default
            final_call_args[p_name] = p_obj.default
        # If still not present and required, factory call will error, which is correct.
        
    # 3. Modify the desired argument
    if arg_name_to_change not in final_call_args and arg_name_to_change not in factory_signature_params:
        raise ValueError(
            f"Argument '{arg_name_to_change}' is not a valid argument "
            f"for the factory method {factory_create_method.__name__}."
            f"Available args: {list(factory_signature_params.keys())}"
        )
    final_call_args[arg_name_to_change] = new_value
    
    # 4. Call the factory
    #print(f"Calling factory with args: {final_call_args}")
    new_model = factory_create_method(**final_call_args)
    
    return new_model


def slice_eta_flat(eta_flat: jnp.ndarray,      # (B*M, L)
                   centers_flat: jnp.ndarray,  # (B*M,)
                   width: int) -> jnp.ndarray:  # → (B*M, 2*width+1)
    """
    For each row in eta_flat, grab the 2*width+1 entries
    centered at centers_flat[i], with wraparound.
    """
    Bm, L = eta_flat.shape

    # offsets = [-width, ..., +width]
    offsets = jnp.arange(-width, width + 1)      # shape (2*width+1,)

    # build an index array of shape (B*M, 2*width+1)
    idx = (centers_flat[:, None] + offsets[None, :]) % L

    # gather
    return jnp.take_along_axis(eta_flat, idx, axis=1)


def _slice_one_sequence(x_l_d: jnp.ndarray,
                        js_m:     jnp.ndarray,
                        *,
                        width: int) -> jnp.ndarray:
    """
    x_l_d  : (L,  D)
    js_m   : (M,)          integer centres (any int32/64 dtype)
    returns: (M, 2*width+1, D)
    """
    L        = x_l_d.shape[0]                      # static at trace-time
    offsets  = jnp.arange(-width, width + 1)       # (L′,)  Python constant
    Lprime   = offsets.size                        # = 2*width+1

    def gather_for_j(j_scalar):
        idx = (offsets + j_scalar) % L             # wrap-around indices
        return x_l_d[idx]                          # (L′, D)

    return jax.vmap(gather_for_j)(js_m)            # (M, L′, D)


# ----------------------------------------------------------------------
# 2.  VMAP OVER BATCH  →  (B, M, L′, D)   then flatten
# ----------------------------------------------------------------------

def _remap_and_clip_multi(x_b_l_d: jnp.ndarray,
                          js_b_m:  jnp.ndarray,
                          *,
                          width: int) -> jnp.ndarray:
    """
    x_b_l_d : (B, L, D)
    js_b_m  : (B, M)
    returns : (B*M, 2*width+1, D)
    """
    # inner_fn is (L,D) × (M,) → (M,L′,D) with width fixed
    inner_fn = partial(_slice_one_sequence, width=width)

    # vmap over batch  →  (B, M, L′, D)
    windows = jax.vmap(inner_fn, in_axes=(0, 0))(x_b_l_d, js_b_m)

    # flatten first two axes
    B, M, Lprime, D = windows.shape
    return windows.reshape(B * M, Lprime, D)


# ----------------------------------------------------------------------
# 3.  T R E E  W R A P P E R
# ----------------------------------------------------------------------

def clip_activations(tree, js, *, width: int):
    """
    tree : arbitrary pytree whose leaves may include 3-D arrays (B, L, D)
    js   :  
            • jnp.ndarray[int] shaped (B, M)
    width: half-window size (Python int)
    """

    # ------------------------------------------------------------------
    def _clip_leaf(arr):
        # if it's not a jnp.ndarray of rank ≥3, leave it untouched
        if not hasattr(arr, "ndim") or arr.ndim < 3:
            return arr

        # handle exactly (B, L, D)—your original path
        if arr.ndim == 3:
            B, L, D = arr.shape
            # make js shape (B, M) if it was a scalar
            js_b_m = js if js.ndim == 2 else jnp.full((B,1), int(js), dtype=jnp.int32)
            return _remap_and_clip_multi(arr, js_b_m, width=width)

        # any higher-rank: treat the last two dims as (L, D), everything else as 'batch dims'
        b, *otherdims, L, D = arr.shape
        # flatten all the leading dims into one
        Bflat = b*math.prod(otherdims)
        arr_flat = arr.reshape((Bflat, L, D))

        # repeat js for those extra dims.
        # js has shape (B, M) where B == batch_dims[0].
        # we must tile each js[b] exactly (Bflat // B) times.
        B, M = js.shape
        factor = Bflat // B
        js_flat = jnp.repeat(js, repeats=factor, axis=0)  # shape (Bflat, M)

        # now apply exactly the same 3-D clip
        windows_flat = _remap_and_clip_multi(arr_flat, js_flat, width=width)

        # windows_flat has shape (Bflat * M, L′, D)

        # reshape back:
        Lp = 2*width + 1
        out = windows_flat.reshape(b, *otherdims, M, Lp, D) 

        # Reshape out to (b*m, otherdims, Lp, )
        out= out.reshape(-1, *otherdims, Lp, D)
        return out


    # pytree-aware traversal (works inside/​outside jit)
    return jax.tree_util.tree_map(
        _clip_leaf,
        tree,
    )

def tile_leaves(tree, *, factor: int):
    """
    tree : arbitrary pytree whose leaves may include 3-D arrays (B, L, D)
    js   :  
            • jnp.ndarray[int] shaped (B, M)
    width: half-window size (Python int)
    """

    # ------------------------------------------------------------------
    def _tile_leaf(arr):
        # if it's not a jnp.ndarray of rank ≥3, leave it untouched
        if not hasattr(arr, "ndim") or arr.ndim != 2:
            return arr

        return jnp.repeat(arr, repeats=factor, axis=0)




    # pytree-aware traversal (works inside/​outside jit)
    return jax.tree_util.tree_map(
        _tile_leaf,
        tree,
    )


def get_idx(sections):
    positions = jnp.arange(sections[-1])
    return jnp.searchsorted( sections,positions, side='right') 


def get_centers(eta, sigma, idx):
    delta = eta - sigma[idx,...]
    return jnp.argmax(delta != 0, axis = 1)


def build_flip_sites(sigma, operator):
    """
    sigma    : (B, L)    samples fed to the operator
    operator : any NetKet operator with get_conn_flattened
    returns
        eta        : (B*M, L)
        mels       : (B*M,)
        flip_site  : (B, M)   – centres correctly aligned with eta
    """
    B, L = sigma.shape
    sections = jnp.arange(B)
    eta, mels = operator.get_conn_flattened(
        sigma,     # same call you already had
        sections=sections          # <-- ask for the offsets
    )

    # For every row in `eta` tell us which batch entry it came from
    batch_for_row = jnp.repeat(jnp.arange(B), jnp.diff(sections))   # (B*M,)

    # Duplicate the right σ-row next to every η-row
    sigma_rep = sigma[batch_for_row]                                # (B*M, L)

    diff          = eta - sigma_rep
    centers_flat  = jnp.argmax(diff != 0, axis=1)                   # (B*M,)

    # If every sample has the same number of connections M
    M = eta.shape[0] // B
    flip_site = centers_flat.reshape(B, M)                          # (B, M)

    return eta, mels, flip_site




@partial(jax.jit, static_argnames=('N', ))
def build_powerlaw_kernel(N: int, alpha: float) -> jnp.ndarray:
    """
    Real-space coupling kernel of length N for a periodic chain:
      K[r] = 1/r^alpha  for r=1,...,N//2
           = 0          for r=0
           = K[N-r]     for r>N/2 (by symmetry)
    """
    r_indices = jnp.arange(N)
    # Distances 0,1,2,...,N-1 mapped to periodic |r|
    # For r_idx, periodic distance is min(r_idx, N - r_idx)
    # Example N=5: r_indices = [0,1,2,3,4]
    # periodic_r = [min(0,5), min(1,4), min(2,3), min(3,2), min(4,1)]
    #            = [0, 1, 2, 2, 1]
    periodic_r = jnp.maximum(jnp.minimum(r_indices, N - r_indices), 1)

    # Kernel K[r] = 1/|r|^alpha for r > 0, else 0
    kernel_values = jnp.where(periodic_r > 0, (1.0 / periodic_r) ** alpha, 1.0)
    return kernel_values

@functools.lru_cache
def _kernel_fft(N: int, alpha: float):
    k = build_powerlaw_kernel(N, alpha).astype(jnp.float32)
    k = jnp.fft.rfft(k)
    
    return k       # cache real-FFT once

@partial(jax.jit, static_argnames=("alpha", "width", "token_size"))
def zz_matrix_elements(
    sigma: jnp.ndarray,  # shape (b, N) or (N,), entries ±1
    J: float,
    alpha: float,
    width : int  = 0,
    token_size: int = 1
) -> jnp.ndarray: # Returns mels, shape (b,)
    """
    Compute matrix elements = ½ ∑_{i≠j} J_eff * σ_i σ_j / |i-j|^α
    by convolution: E_zz = ½ J_eff * σ · (K * σ).
    J_eff is J normalized by (sum_of_kernel_elements_excluding_K0 * N).
    """
    if sigma.ndim == 1: # Handle single configuration
        sigma = sigma[jnp.newaxis, :]
    b, N = sigma.shape # N is a static Python int here

    if N <= 1: # For N=0 or N=1, interaction energy is 0
        return jnp.zeros(b, dtype=sigma.dtype) # Ensure dtype matches sigma

    # For N > 1:
    K_vals = build_powerlaw_kernel(N, alpha) # K_vals[0] is 0.0.
    Kf = jnp.fft.rfft(K_vals) # FFT of the kernel

    kernel_sum_norm = K_vals[0:].sum()
    J_eff = J / (kernel_sum_norm * N)

    # FFT both arrays, multiply, and inverse-FFT
    σf = jnp.fft.rfft(sigma, axis=1)
    #Kf = _kernel_fft(N, alpha) # Cached FFT of kernel
    #Kf = jnp.array(Kf)

    # Convolution term: (K * σ)
    convolved_term = jnp.fft.irfft(σf * Kf[jnp.newaxis, ...]).real

    # Dot into sigma and multiply by ½J_eff
    mels = 0.5 * J_eff * jnp.sum(sigma * convolved_term, axis=1)

    indices = jnp.arange(-token_size * width, token_size*width + token_size)
    #sigma = jax.lax.dyna mic_slice_in_dim(sigma, 0, 2 * width + 1, axis = 1) 
    sigma = sigma[:, indices]
    lengths = jnp.ones(sigma.shape[0]) 
    sites = jnp.zeros(sigma.shape[0], dtype = jnp.int32)
        

    return sigma, mels, lengths, sites 



@partial(jax.jit, static_argnames=( "width", "token_size"))
def z_matrix_elements(
    sigma: jnp.ndarray,  # shape (b, N) or (N,), entries ±1
    hz: float,
    width : int  = 0,
    token_size: int = 1
) -> jnp.ndarray: # Returns mels, shape (b,)
    """
    Compute matrix elements = ½ ∑_{i≠j} J_eff * σ_i σ_j / |i-j|^α
    by convolution: E_zz = ½ J_eff * σ · (K * σ).
    J_eff is J normalized by (sum_of_kernel_elements_excluding_K0 * N).
    """
    if sigma.ndim == 1: # Handle single configuration
        sigma = sigma[jnp.newaxis, :]
    b, N = sigma.shape # N is a static Python int here

    if N <= 1: # For N=0 or N=1, interaction energy is 0
        return jnp.zeros(b, dtype=sigma.dtype) # Ensure dtype matches sigma

    mels = hz * jnp.sum(sigma, axis=1)

    indices = jnp.arange(-token_size * width, token_size*width + token_size)
    #sigma = jax.lax.dyna mic_slice_in_dim(sigma, 0, 2 * width + 1, axis = 1) 
    sigma = sigma[:, indices]
    lengths = jnp.ones(sigma.shape[0])
    sites = jnp.zeros(sigma.shape[0], dtype=jnp.int32)

    return sigma, mels, lengths, sites 



@partial(jax.jit, static_argnames=('width', 'token_size'))
def x_matrix_elements(sigma: jnp.ndarray,
                      width: int,
                      strength: float = -1.0,
                      *,
                      token_size: int):
    """
    σˣ operator, but the network’s receptive field is defined in *tokens*
    (blocks of `token_size` spins).
    Returned `eta` already carries the clipped windows the model expects.
    """
    B, L = sigma.shape
    if L % token_size != 0:
        raise ValueError("L must be divisible by token_size")

    # ------------------------------------------------------------------
    # 1) enumerate every spin in every batch sample
    batches = jnp.arange(B)
    idx     = jnp.repeat(batches, L)          # (B*L,)
    sites   = jnp.arange(B*L) % L             # absolute spin index

    # 2) token centres for the partial-eval path
    center_tokens = sites // token_size       # (B*L,)

    centers_spin_space = token_size * center_tokens
    # 3) slice (token aware)
    eta = slice_tokens_jit(sigma, idx, centers_spin_space, width=width,  token_size=token_size)

    # 4) flip the *single* spin that is acted on by σˣ
    in_patch_offset      = sites % token_size
    center_offset_in_win = width * token_size + in_patch_offset

    # Create row indices to pair with center_offset_in_win
    row_indices = jnp.arange(eta.shape[0])
    eta = eta.at[row_indices, center_offset_in_win].set(
        -eta[row_indices, center_offset_in_win])

    # 5) remaining bookkeeping exactly as before
    mels    = strength * jnp.ones(eta.shape[0]) / L
    lengths = L * jnp.ones(B, dtype=jnp.int32)      # each sample has L connections
    flip_site = center_tokens                       # what the partial net needs

    return eta, mels, lengths, sites # Changing this warning! 


# Your original compose_idx, with minor robustness improvements
def compose_idx(start, stop):
    lengths = stop - start
    max_len = jnp.max(lengths)
    
    offsets = jnp.arange(max_len)
    all_idx = start[:, None] + offsets[None, :]
    mask = offsets[None, :] < lengths[:, None]
    merged_idx = all_idx[mask]
    return merged_idx



def get_idx_sections_from_lengths(lengths_A, lengths_B):

    all_segment_lengths =  jnp.stack([lengths_A, lengths_B], axis=1).ravel()
        
    # Calculate cumulative sums to get stop points for each segment
    cumulative_stops = jnp.cumsum(all_segment_lengths)
    
    # Start points are [0, L_A0, L_A0+L_B0, L_A0+L_B0+L_A1, ...]
    all_starts = jnp.concatenate([jnp.array([0]), cumulative_stops[:-1]])
    
    # Extract starts and stops for A and B segments
    # A segments are at even indices (0, 2, 4, ...) in all_starts/all_stops
    # B segments are at odd indices (1, 3, 5, ...)
    start_A = all_starts[0::2] # Starts of A0, A1, ...
    stop_A = cumulative_stops[0::2]  # Stops of A0, A1, ...
    
    start_B = all_starts[1::2] # Starts of B0, B1, ...
    stop_B = cumulative_stops[1::2]  # Stops of B0, B1, ...
    
    merged_idx_A = compose_idx(start_A, stop_A)
    merged_idx_B = compose_idx(start_B, stop_B)
    
    return merged_idx_A, merged_idx_B


def merge_arrays_by_section(arrayA, arrayB, lengths_A, lengths_B, axis = 0):
    lengths_A = jnp.array(lengths_A, dtype=jnp.int32)
    lengths_B = jnp.array(lengths_B, dtype=jnp.int32)

    idx_A, idx_B = get_idx_sections_from_lengths(lengths_A, lengths_B)

    # allocate once and scatter both A and B in a single update
    total_rows = arrayA.shape[0] + arrayB.shape[0]
    newArray = jnp.zeros((total_rows,) + arrayA.shape[1:], dtype=arrayA.dtype)
    combined_idx = jnp.concatenate([idx_A, idx_B], axis=0)
    combined_vals = jnp.concatenate([arrayA, arrayB], axis=0)
    newArray = newArray.at[combined_idx].set(combined_vals)
    return newArray, lengths_A + lengths_B


def _embed_windows_into_full(
    sigma_full: jnp.ndarray,
    eta_win: jnp.ndarray,
    lengths: jnp.ndarray,
    sites: jnp.ndarray,
    *,
    width: int,
    token_size: int,
) -> jnp.ndarray:
    """
    Expand windowed configurations `eta_win` back into full-length states.
    """
    lengths = jnp.asarray(lengths, dtype=jnp.int32)
    sites = jnp.asarray(sites, dtype=jnp.int32)
    sigma_full = jnp.asarray(sigma_full)

    L = sigma_full.shape[-1]
    total_rows = eta_win.shape[0]

    if lengths.size == 0:
        return jnp.zeros((total_rows, L), dtype=sigma_full.dtype)

    sections = jnp.cumsum(lengths)
    idx = get_idx(sections)

    base = sigma_full[idx, :]

    window_len = eta_win.shape[1]
    offsets = jnp.arange(window_len, dtype=jnp.int32) - width * token_size

    center_tokens = sites // token_size
    centers_spin_space = token_size * center_tokens
    positions = (centers_spin_space[:, None] + offsets[None, :]) % L

    row_idx = jnp.arange(total_rows, dtype=jnp.int32)[:, None]
    eta_full = base.at[row_idx, positions].set(eta_win)
    return eta_full

class OperatorMatrixElements: 

    def __init__(self, op_func, width = 0):
        
        self.funcs = [lambda x : op_func(x, width)]
        self.width = width

    def apply_operator(self, sigma):
        sigma.astype(jnp.float32)
        eta, mels, sections, sites = self.funcs[0](sigma)

        for j in range(1, len(self.funcs)):
            eta_plus, mels_plus, sections_plus, sites_plus = self.funcs[j](sigma)
            eta, _ = merge_arrays_by_section(eta, eta_plus, sections, sections_plus)
            mels, _ = merge_arrays_by_section(mels, mels_plus, sections, sections_plus)
            sites, sections = merge_arrays_by_section(sites, sites_plus, sections, sections_plus)

        return eta, mels, sections, sites

    def __add__(self, other: "OperatorMatrixElements") -> "OperatorMatrixElements":
        if not isinstance(other, OperatorMatrixElements):
            return NotImplemented
        # create a new instance preserving width and combining funcs
        new = OperatorMatrixElements(self.funcs[0], self.width)
        new.funcs = self.funcs + other.funcs
        return new


def apply_operator_power(
    operator: OperatorMatrixElements,
    sigma: jnp.ndarray,
    power: int,
    *,
    width: int,
    token_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Apply `operator` repeatedly `power` times, returning the accumulated
    configurations, matrix elements, and connection counts.
    """
    if power < 0:
        raise ValueError("power must be non-negative")

    sigma = jnp.asarray(sigma)
    batch_size = sigma.shape[0]

    current_sigma = sigma
    mels = jnp.ones((batch_size,), dtype=sigma.dtype)
    origins = jnp.arange(batch_size, dtype=jnp.int32)

    if power == 0:
        lengths_eff = jnp.ones((batch_size,), dtype=jnp.int32)
        return current_sigma, mels, lengths_eff

    for _ in range(power):
        eta_win, mels_step, lengths, sites = operator.apply_operator(current_sigma)
        lengths = jnp.asarray(lengths, dtype=jnp.int32)

        current_sigma = _embed_windows_into_full(
            current_sigma,
            eta_win,
            lengths,
            sites,
            width=width,
            token_size=token_size,
        )

        mels = jnp.repeat(mels, lengths) * mels_step
        origins = jnp.repeat(origins, lengths)

    lengths_eff = jnp.bincount(origins, minlength=batch_size).astype(jnp.int32)

    return current_sigma, mels, lengths_eff

def hamiltonian_tfim_matrix_element(J, alpha, width, strength : int = -1.0, hz : float = 0.0, token_size : int = 1):
    sigma_zz_func = lambda s, w : zz_matrix_elements(s, J, alpha, w, token_size=token_size)
    sigma_x_func = lambda s, w : x_matrix_elements(s, w, strength, token_size=token_size)
    
    op = OperatorMatrixElements(sigma_zz_func, width) + OperatorMatrixElements(sigma_x_func, width)
    
    if hz != 0.0:
        sigma_z_func = lambda s, w : z_matrix_elements(s, hz, w, token_size=token_size)
        op += OperatorMatrixElements(sigma_z_func, width)
    
    return op

def hamiltonian_tfim(hilbert, J, alpha, width, strength : int = -1.0, hz : float = 0.0, token_size : int = 1, z2_project = False):
    mat_elements = hamiltonian_tfim_matrix_element(J, alpha, width, strength, hz=hz, token_size=token_size)
    op = TurboTurboOperator(
        hilbert=hilbert,
        operator=mat_elements,
        width=width,
        fast_operator=True, 
        z2_project=z2_project,
    )
    
    return op

@partial(jax.jit, static_argnames=("width", "token_size"))
def j1j2_zz_matrix_elements(
    sigma: jnp.ndarray,
    J1_z: float,
    J2_z: float,
    width: int = 0,
    token_size: int = 1,
):
    sigma = jnp.asarray(sigma)
    if sigma.ndim == 1:
        sigma = sigma[jnp.newaxis, :]

    B, L = sigma.shape
    J1_z_arr = jnp.asarray(J1_z)
    J2_z_arr = jnp.asarray(J2_z)
    mel_dtype = jnp.result_type(J1_z_arr, J2_z_arr, jnp.float32)
    J1_z_arr = J1_z_arr.astype(mel_dtype)
    J2_z_arr = J2_z_arr.astype(mel_dtype)

    if L == 0:
        mels = jnp.zeros((B,), dtype=mel_dtype)
    else:
        sigma_float = sigma.astype(mel_dtype)
        nn = jnp.sum(sigma_float * jnp.roll(sigma_float, -1, axis=1), axis=1)
        nnn = jnp.sum(sigma_float * jnp.roll(sigma_float, -2, axis=1), axis=1)
        mels = J1_z_arr * nn + J2_z_arr * nnn

    eta_win, lengths, sites = get_diagonal_args(sigma, width, token_size=token_size)
    return eta_win, mels, lengths, sites


@partial(jax.jit, static_argnames=("width", "token_size"))
def j1j2_xy_matrix_elements(
    sigma: jnp.ndarray,
    J1_xy: float,
    J2_xy: float,
    width: int = 0,
    token_size: int = 1,
):
    sigma = jnp.asarray(sigma)
    if sigma.ndim == 1:
        sigma = sigma[jnp.newaxis, :]
    B, L = sigma.shape

    if L % token_size != 0:
        raise ValueError("L must be divisible by token_size")

    window_len = token_size * (2 * width + 1)
    max_distance = window_len - 1
    if max_distance < 1:
        raise ValueError(
            "Window width/token_size combination must include nearest neighbours for XY terms."
        )
    offsets = jnp.arange(window_len, dtype=jnp.int32) - width * token_size

    J1_xy_arr = jnp.asarray(J1_xy)
    J2_xy_arr = jnp.asarray(J2_xy)
    mel_dtype = jnp.result_type(J1_xy_arr, J2_xy_arr, jnp.float32)
    J1_xy_arr = J1_xy_arr.astype(mel_dtype)
    J2_xy_arr = J2_xy_arr.astype(mel_dtype)

    if L < 2 or B == 0:
        eta = jnp.zeros((0, window_len), dtype=sigma.dtype)
        mels = jnp.zeros((0,), dtype=mel_dtype)
        lengths = jnp.zeros((B,), dtype=jnp.int32)
        sites = jnp.zeros((0,), dtype=jnp.int32)
        return eta, mels, lengths, sites

    site_indices = jnp.arange(L, dtype=jnp.int32)
    batches = jnp.arange(B, dtype=jnp.int32)
    idx = jnp.repeat(batches, L)
    left_sites = jnp.tile(site_indices, B)

    def build_term(delta, coupling):
        if delta > max_distance:
            raise ValueError(
                f"Window width/token_size={token_size}*{width} cannot support bonds separated by {delta} sites."
            )
        total = B * L
        right_sites = (left_sites + delta) % L
        center_sites = (left_sites + (delta // 2)) % L
        center_tokens = center_sites // token_size
        centers_spin_space = token_size * center_tokens
        windows = slice_tokens_jit(
            sigma,
            idx,
            centers_spin_space,
            width=width,
            token_size=token_size,
        )

        spins_left = sigma[idx, left_sites]
        spins_right = sigma[idx, right_sites]
        valid = spins_left != spins_right
        coupling_mask = coupling != 0.0
        flip_mask = jnp.logical_and(valid, coupling_mask)

        window_positions = (centers_spin_space[:, None] + offsets[None, :]) % L
        left_mask = jnp.logical_and(flip_mask[:, None], window_positions == left_sites[:, None])
        right_mask = jnp.logical_and(flip_mask[:, None], window_positions == right_sites[:, None])
        flipped_left = -spins_left
        flipped_right = -spins_right
        windows = jnp.where(left_mask, flipped_left[:, None], windows)
        windows = jnp.where(right_mask, flipped_right[:, None], windows)

        mels_term = (2.0 * coupling) * flip_mask.astype(mel_dtype)
        return windows, mels_term, center_sites

    eta_terms = []
    mels_terms = []
    sites_terms = []
    term_count = 0

    eta_j1, mels_j1, sites_j1 = build_term(1, J1_xy_arr)
    eta_terms.append(eta_j1.reshape(B, L, window_len))
    mels_terms.append(mels_j1.reshape(B, L))
    sites_terms.append(sites_j1.reshape(B, L))
    term_count += 1

    if L >= 3:
        eta_j2, mels_j2, sites_j2 = build_term(2, J2_xy_arr)
        eta_terms.append(eta_j2.reshape(B, L, window_len))
        mels_terms.append(mels_j2.reshape(B, L))
        sites_terms.append(sites_j2.reshape(B, L))
        term_count += 1

    eta = jnp.concatenate(eta_terms, axis=1).reshape(B * term_count * L, window_len)
    mels = jnp.concatenate(mels_terms, axis=1).reshape(B * term_count * L)
    sites = jnp.concatenate(sites_terms, axis=1).reshape(B * term_count * L)

    lengths = (term_count * L) * jnp.ones((B,), dtype=jnp.int32)

    return eta, mels, lengths, sites


def hamiltonian_j1j2_matrix_element(
    J1_xy: float,
    J1_z: float,
    J2_xy: float,
    J2_z: float,
    width: int,
    hz: float = 0.0,
    token_size: int = 1,
):
    token_size = int(token_size)
    J1_xy_val = float(J1_xy)
    J2_xy_val = float(J2_xy)
    J1_z_val = float(J1_z)
    J2_z_val = float(J2_z)
    hz_val = float(hz)

    max_distance = token_size * (2 * width + 1) - 1
    if max_distance < 1 and J1_xy_val != 0.0:
        raise ValueError(
            "window is too narrow to include nearest-neighbour XY terms."
        )
    if max_distance < 2 and J2_xy_val != 0.0:
        raise ValueError(
            "window is too narrow to include next-nearest-neighbour XY terms."
        )

    sigma_zz_func = lambda s, w: j1j2_zz_matrix_elements(
        s, J1_z_val, J2_z_val, w, token_size=token_size
    )
    op = OperatorMatrixElements(sigma_zz_func, width)

    if (J1_xy_val != 0.0) or (J2_xy_val != 0.0):
        sigma_xy_func = lambda s, w: j1j2_xy_matrix_elements(
            s, J1_xy_val, J2_xy_val, w, token_size=token_size
        )
        op += OperatorMatrixElements(sigma_xy_func, width)

    if hz_val != 0.0:
        sigma_z_func = lambda s, w: z_matrix_elements(
            s, hz_val, w, token_size=token_size
        )
        op += OperatorMatrixElements(sigma_z_func, width)

    return op


def hamiltonian_j1j2(
    hilbert,
    J1_xy: float,
    J1_z: float,
    J2_xy: float,
    J2_z: float,
    width: int,
    hz: float = 0.0,
    token_size: int = 1,
    z2_project: bool = False,
):
    mat_elements = hamiltonian_j1j2_matrix_element(
        J1_xy,
        J1_z,
        J2_xy,
        J2_z,
        width,
        hz=hz,
        token_size=token_size,
    )
    op = TurboTurboOperator(
        hilbert=hilbert,
        operator=mat_elements,
        width=width,
        fast_operator=True,
        z2_project=z2_project,
    )
    return op

# ---------------------------------------------------------------------
# the operator class ---------------------------------------------------
# ---------------------------------------------------------------------
class TurboTurboOperator(AbstractOperator):
    """
    A one-body spin-flip operator that re-evaluates the network using
    clipped intermediates.  It is hermitian if the underlying matrix
    elements M_k are real (default: 1).
    """


    def __init__(self, hilbert, operator, *, width, L=None, fast_operator= False, fused_kernel = False, z2_project = False):
        super().__init__(hilbert)
        self._width = int(width)
        self._L = int(L if L is not None else hilbert.size)
        self.operator = operator
        self.z2_project = z2_project
        self.fused_kernel = fused_kernel
        if isinstance(operator, OperatorMatrixElements) or fast_operator:
            self.fast_operator = True
        else:
            self.fast_operator = False

    # ---- mandatory abstract-operator properties ---------------------
    @property
    def dtype(self):
        return float

    @property
    def is_hermitian(self):
        return True

    # ---- NetKet “lean” interface  -----------------------------------
    # tell NetKet *which* kernel to use ...
    @nk.vqs.get_local_kernel.dispatch
    def _get_kernel(vstate : nk.vqs.VariationalState, op : AbstractOperator, chunk_size : Optional[int] = None):
        model = vstate._model

        
        # Adjust teh width if it is not right
        if model.s4_l_width != op._width:
            model = duplicate_model_via_factory(
                original_model=model,
                arg_name_to_change="s4_l_width",
                new_value=op._width,
                factory_create_method=mnqs.DysonNetFactory.create_model
            )
        
        z2_project = op.z2_project if hasattr(op, 'z2_project') else False
        shift_project = model.shift_project if hasattr(model, 'shift_project') else False

        # 1) Pre-partial out your full and partial apply_funs, with all
        full_apply = nk.utils.HashablePartial(
            model.apply,
            mode="full",
            cache_jac=True,
            mutable=["intermediates", "cache"],
            capture_intermediates=False,
            remove_shift=True, 
        )
        partial_apply = nk.utils.HashablePartial(
            model.apply,
            mode="partial",
            mutable=["intermediates", "cache"],
            capture_intermediates=False,
            remove_shift=True,
        )

        @jax.jit
        def small_kernel(params, sigma, eta, centers, mels): 
            """
            This is the kernel function that will be called by NetKet.
            It takes the parameters, sigma, eta and centers as input
            and returns the kernel value.
            """
            b, l = sigma.shape
            idx, c = centers
            log_psi_sigma, acts_sigma = full_apply(params, sigma)


            # WARNING: If something should break it probably has to do with adding cache
            log_psi_eta, _ = partial_apply(
                {"intermediates": acts_sigma["intermediates"], "cache" : acts_sigma["cache"], **params}, 
                eta, 
                centers = centers
            )

            if z2_project:
                log_psi_sigma_flipped, acts_sigma_flip = full_apply(params, -sigma)

                log_psi_eta_flipped, _ = partial_apply(
                    {"intermediates": acts_sigma_flip["intermediates"], "cache" : acts_sigma_flip["cache"], **params}, 
                    -eta, 
                    centers = centers
                )

                log_psi_sigma = jnp.logaddexp(log_psi_sigma, log_psi_sigma_flipped) - jnp.log(2)
                log_psi_eta = jnp.logaddexp(log_psi_eta, log_psi_eta_flipped) - jnp.log(2)

            if shift_project: 
                token_size = int(getattr(model, "token_size", 1))
                sigma_shifted, eta_shifted = shift_arrays(
                    eta,
                    sigma,
                    idx,
                    c,
                    width=int(op._width),
                    token_size=token_size,
                )

                c_shifted = (c + 1) % l
                centers_shifted = (idx, c_shifted)

                log_psi_sigma_shifted, acts_sigma_shifted = full_apply(params, sigma_shifted)

                log_psi_eta_shifted, _ = partial_apply(
                    {"intermediates": acts_sigma_shifted["intermediates"], "cache" : acts_sigma_shifted["cache"], **params}, 
                    eta_shifted, 
                    centers = centers_shifted
                )

                log_psi_sigma = jnp.logaddexp(log_psi_sigma, log_psi_sigma_shifted) - jnp.log(2)
                log_psi_eta = jnp.logaddexp(log_psi_eta, log_psi_eta_shifted) - jnp.log(2)

            res = mels * jnp.exp(log_psi_eta - log_psi_sigma[idx, ...])
            return jnp.sum(res.reshape(b, -1), axis=1)
            

        @partial(jax.jit, static_argnames=('chunk_size'))
        def small_kernel_chunked(params, sigma, eta, centers, mels, chunk_size = None):
            # 1) Chunk all inputs up front
            sigma_chunks = sigma.reshape((-1, chunk_size, sigma.shape[-1]))

            larger_chunk_size    = chunk_size * (eta.shape[0] // sigma.shape[0])
            eta_chunks        = eta.reshape((-1, larger_chunk_size, eta.shape[-1]))
            idx, c = centers
            idx = idx.reshape((-1, larger_chunk_size))
            c = c.reshape((-1, larger_chunk_size))
            centers_chunks = (idx, c)
            mels_chunks       = mels.reshape((-1, larger_chunk_size))
        
            # 2) Define the scan body: takes carry (unused) + one slice of each input
            @jax.jit
            def body(carry, elems):
                σ_chunk, η_chunk, (idx_chunk, c_chunk), mel_chunk = elems
        
                # Re-normalize idx so it's non-negative per chunk:
                idx_chunk = idx_chunk - jnp.min(idx_chunk, keepdims=True)
                centers_chunk = (idx_chunk, c_chunk)
        
                # Call your small_kernel on this one chunk
                out_chunk = small_kernel(params, σ_chunk, η_chunk, centers_chunk, mel_chunk)
        
                # We don’t need a carry, so return None
                return None, out_chunk
        
            # 3) Scan over the tuple of chunked inputs
            _, out_chunks = jax.lax.scan(
                body,
                init=None,
                xs=(sigma_chunks, eta_chunks, centers_chunks, mels_chunks),
            )
        
            # 4) Un-chunk the stacked output back to original shape
            return out_chunks.reshape(-1)


        def kernel(_ignored, params, sigma, extra_args, chunk_size=None):   
            eta, mels, centers = extra_args

            # 2) Call your two pre-made apply_funs—no kwargs passed at call time!
            if "params" not in params:
                params = {"params": params}
            
            if chunk_size is None:
                return small_kernel(params, sigma, eta, centers, mels)
            else:
                return small_kernel_chunked(params, sigma, eta, centers, mels, chunk_size=chunk_size)


        # 3) Finally wrap your kernel itself in a HashablePartial so that
        #    those two apply_funs get captured statically, too.
        return nk.utils.HashablePartial(kernel)


    # ... and *what* extra arguments the kernel needs.
    @nk.vqs.get_local_kernel_arguments.dispatch
    def _get_kernel_args(vstate : nk.vqs.VariationalState, self : AbstractOperator, chunk_size : Optional[int] = None, standard : bool = False):
        """
        Builds (samples, extra_args) where
            samples     : vstate.samples  (B, N)
            extra_args  : tuple to feed to the kernel
                          (eta, mels, flip_site)
        """
        
        sigma = vstate.samples
        sigma = sigma.reshape((-1, sigma.shape[-1]))

        if self.fast_operator and not standard:
            eta, mels, lengths, flip_site = self.operator.apply_operator(sigma)
            idx = get_idx(jnp.cumsum(lengths))
        else:
            n_conn = self.operator.n_conn(sigma)                # shape (B,), number of connections per sample
            sections = jnp.cumsum(n_conn)         # shape (B,), cumulative end‐points
            eta, mels = self.operator.get_conn_flattened(sigma, sections)

            idx = get_idx(sections)
            flip_site = get_centers(eta, sigma, idx) 
            #eta = slice_eta_flat(eta, flip_site, width = self._width)  # (B*M, 2*width+1, L) 

        return sigma, (eta, mels, (idx, flip_site))
    

    def _get_sigma_args(sigma, self, chunk_size : Optional[int] = None, standard = False):
        """
        Builds (samples, extra_args) where
            samples     : vstate.samples  (B, N)
            extra_args  : tuple to feed to the kernel
                          (eta, mels, flip_site)
        """
        
        sigma = sigma.reshape((-1, sigma.shape[-1]))

        if self.fast_operator and not standard:
            eta, mels, lengths, flip_site = self.operator.apply_operator(sigma)
            idx = get_idx(jnp.cumsum(lengths))
        else:
            n_conn = self.operator.n_conn(sigma)                # shape (B,), number of connections per sample
            sections = np.cumsum(n_conn)         # shape (B,), cumulative end‐points
            eta, mels = self.operator.get_conn_flattened(sigma, sections)

            idx = get_idx(sections)
            flip_site = get_centers(eta, sigma, idx) 
            #eta = slice_eta_flat(eta, flip_site, width = self._width)  # (B*M, 2*width+1, L) 

        return sigma, (eta, mels, (idx, flip_site))
    

    def _get__standard_kernel(vstate, op):

        model = vstate._model
        full_no_jac = nk.utils.HashablePartial(
            model.apply,
            mode="full",
            cache_jac=False,
        )


        def standard_kernel(wavefunc, params, sigma, extra_args):
            eta, mels = extra_args
            b, l = sigma.shape
            m = eta.shape[0] // b
            # 2) Call your two pre-made apply_funs—no kwargs passed at call time!
            log_psi_sigma = full_no_jac({"params" : params}, sigma)
            log_psi_sigma = jnp.repeat(log_psi_sigma, m)
            
            log_psi_eta= full_no_jac({"params" : params}, eta)
            delta = log_psi_eta - log_psi_sigma

            res = mels * jnp.exp(delta )
            res = jnp.sum(res.reshape(b, -1), axis=1)
            return res

        # 3) Finally wrap your kernel itself in a HashablePartial so that
        #    those two apply_funs get captured statically, too.
        return nk.utils.HashablePartial(standard_kernel)
    
    def _get_fused_kernel(vstate, op):

        model = vstate._model


        # Adjust teh width if it is not right
        if model.s4_l_width != op._width:
            model = duplicate_model_via_factory(
                original_model=model,
                arg_name_to_change="s4_l_width",
                new_value=op._width,
                factory_create_method=mnqs.DysonNetFactory.create_model
            )
        

        # 1) Pre-partial out your full and partial apply_funs, with all
        fused_apply = nk.utils.HashablePartial(
            model.apply,
            mutable=["intermediates", "cache"],
            capture_intermediates=False,
            method=model._partial_call
        )



        @jax.jit
        def small_kernel(params, sigma, eta, centers, mels): 
            """
            This is the kernel function that will be called by NetKet.
            It takes the parameters, sigma, eta and centers as input
            and returns the kernel value.
            """
            b, l = sigma.shape
            idx, _ = centers
            (log_psi_sigma, log_psi_eta), _ = fused_apply(params, 
                                                         sigma, 
                                                         eta, 
                                                         centers)

            res = mels * jnp.exp(log_psi_eta - log_psi_sigma[idx, ...])
            return jnp.sum(res.reshape(b, -1), axis=1)
            

        @partial(jax.jit, static_argnames=('chunk_size'))
        def small_kernel_chunked(params, sigma, eta, centers, mels, chunk_size = None):
            # 1) Chunk all inputs up front
            sigma_chunks = sigma.reshape((-1, chunk_size, sigma.shape[-1]))

            larger_chunk_size    = chunk_size * (eta.shape[0] // sigma.shape[0])
            eta_chunks        = eta.reshape((-1, larger_chunk_size, eta.shape[-1]))
            idx, c = centers
            idx = idx.reshape((-1, larger_chunk_size))
            c = c.reshape((-1, larger_chunk_size))
            centers_chunks = (idx, c)
            mels_chunks       = mels.reshape((-1, larger_chunk_size))
        
            # 2) Define the scan body: takes carry (unused) + one slice of each input
            @jax.jit
            def body(carry, elems):
                σ_chunk, η_chunk, (idx_chunk, c_chunk), mel_chunk = elems
        
                # Re-normalize idx so it's non-negative per chunk:
                idx_chunk = idx_chunk - jnp.min(idx_chunk, keepdims=True)
                centers_chunk = (idx_chunk, c_chunk)
        
                # Call your small_kernel on this one chunk
                out_chunk = small_kernel(params, σ_chunk, η_chunk, centers_chunk, mel_chunk)
        
                # We don’t need a carry, so return None
                return None, out_chunk
        
            # 3) Scan over the tuple of chunked inputs
            _, out_chunks = jax.lax.scan(
                body,
                init=None,
                xs=(sigma_chunks, eta_chunks, centers_chunks, mels_chunks),
            )
        
            # 4) Un-chunk the stacked output back to original shape
            return out_chunks.reshape(-1)


        def kernel(_ignored, params, sigma, extra_args, chunk_size=None):   
            eta, mels, centers = extra_args

            # 2) Call your two pre-made apply_funs—no kwargs passed at call time!
            if "params" not in params:
                params = {"params": params}
            
            if chunk_size is None:
                return small_kernel(params, sigma, eta, centers, mels)
            else:
                return small_kernel_chunked(params, sigma, eta, centers, mels, chunk_size=chunk_size)


        # 3) Finally wrap your kernel itself in a HashablePartial so that
        #    those two apply_funs get captured statically, too.
        return nk.utils.HashablePartial(kernel)




# ---------------------------------------------------------------------
# Additional operators ----- --------------------------------
# --------------------------------------------------------------------- 

def get_diagonal_args(sigma, width, token_size=1):
    indices = jnp.arange(-token_size * width, token_size*width + token_size)
    sigma = sigma[:, indices]
    lengths = jnp.ones(sigma.shape[0], dtype=jnp.int32) 
    sites = jnp.zeros(sigma.shape[0], dtype = jnp.int32)

    return sigma, lengths, sites


def zz_correlators(
    sigma: jnp.ndarray,
    *,
    max_r: Optional[int] = None,
) -> jnp.ndarray:
    """
    Compute <sigma_j^z sigma_{j+r}^z> for r=1..max_r for each configuration.
    Returns an array with shape (batch, max_r).
    """
    sigma = jnp.asarray(sigma)
    if sigma.ndim == 1:
        sigma = sigma[jnp.newaxis, :]
    elif sigma.ndim > 2:
        sigma = sigma.reshape(-1, sigma.shape[-1])

    batch, n_sites = sigma.shape
    if max_r is None:
        max_r = n_sites // 2

    rs = jnp.arange(1, max_r + 1, dtype=jnp.int32)

    def _corr(r: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(sigma * jnp.roll(sigma, -r, axis=1), axis=1)

    corrs = jax.vmap(_corr)(rs)  # (max_r, batch)
    return corrs.T.reshape(batch, max_r)


def structure_factor_zz(sigma: jnp.ndarray) -> jnp.ndarray:
    """
    Structure factor S_k = (1/N) |Σ_j sigma_j e^{ikj}|^2 for k_n = 2πn/N.
    Returns array of shape (batch, N) with k ordered as FFT frequencies.
    """
    sigma = jnp.asarray(sigma, dtype=jnp.float32)
    if sigma.ndim == 1:
        sigma = sigma[jnp.newaxis, :]
    elif sigma.ndim > 2:
        sigma = sigma.reshape(-1, sigma.shape[-1])

    n_sites = sigma.shape[-1]
    fft_vals = jnp.fft.fft(sigma, axis=1)
    s_k = (jnp.abs(fft_vals) ** 2) / n_sites
    return jnp.real(s_k)


def magnetization(
    sigma: jnp.ndarray,  # shape (b, N) or (N,), entries ±1
    width : int  = 0,
    token_size: int = 1
): 

    if sigma.ndim == 1: # Handle single configuration
        sigma = sigma[jnp.newaxis, :]
    _, N = sigma.shape # N is a static Python int here

    mels = jnp.sum(sigma, axis = 1) / N
    sigma, lengths, sites = get_diagonal_args(sigma, width, token_size=token_size)

    return sigma, mels, lengths, sites


def magnetization_staggered(
    sigma: jnp.ndarray,  # shape (b, N) or (N,), entries ±1
    width : int  = 0,
    token_size: int = 1
): # Returns mels, shape (b,)

    if sigma.ndim == 1: # Handle single configuration
        sigma = sigma[jnp.newaxis, :]
    b, N = sigma.shape # N is a static Python int here

    factor = (-1) ** (jnp.arange(N))
    mels = jnp.sum(factor[jnp.newaxis, :] * sigma, axis = 1) / N
    sigma, lengths, sites = get_diagonal_args(sigma, width, token_size=token_size)

    return sigma, mels, lengths, sites


def magnetization_sqr(
    sigma: jnp.ndarray,  # shape (b, N) or (N,), entries ±1
    width : int  = 0,
    token_size: int = 1
): # Returns mels, shape (b,)

    if sigma.ndim == 1: # Handle single configuration
        sigma = sigma[jnp.newaxis, :]
    b, N = sigma.shape # N is a static Python int here

    mels = ( jnp.sum(sigma, axis = 1) / N ) ** 2
    sigma, lengths, sites = get_diagonal_args(sigma, width, token_size=token_size)

    return sigma, mels, lengths, sites


def magnetization_staggered_sqr(
    sigma: jnp.ndarray,  # shape (b, N) or (N,), entries ±1
    width : int  = 0,
    token_size: int = 1
): # Returns mels, shape (b,)

    if sigma.ndim == 1: # Handle single configuration
        sigma = sigma[jnp.newaxis, :]
    b, N = sigma.shape # N is a static Python int here


    factor = (-1) ** (jnp.arange(N))
    mels = ( jnp.sum(factor[jnp.newaxis, :] * sigma, axis = 1) / N ) ** 2
    sigma, lengths, sites = get_diagonal_args(sigma, width, token_size=token_size)

    return sigma, mels, lengths, sites


def magnetization_op(hilbert, width, token_size : int = 1, z2_project = False):
    mag_func = lambda s, w : magnetization(s, w, token_size=token_size)
    mat_elements = OperatorMatrixElements(mag_func, width)

    op = TurboTurboOperator(
        hilbert=hilbert,
        operator=mat_elements,
        width=width,
        fast_operator=True, 
        z2_project=z2_project,
    )

    return op


def magnetization_sqr_op(hilbert, width, token_size : int = 1, z2_project = False):
    mag_func = lambda s, w : magnetization_sqr(s, w, token_size=token_size)
    mat_elements = OperatorMatrixElements(mag_func, width)

    op = TurboTurboOperator(
        hilbert=hilbert,
        operator=mat_elements,
        width=width,
        fast_operator=True,
        z2_project=z2_project
    )

    return op

def staggered_magnetization_op(hilbert, width, token_size : int = 1, z2_project = False):
    mag_func = lambda s, w : magnetization_staggered(s, w, token_size=token_size)
    mat_elements = OperatorMatrixElements(mag_func, width)

    op = TurboTurboOperator(
        hilbert=hilbert,
        operator=mat_elements,
        width=width,
        fast_operator=True,
        z2_project=z2_project
    )

    return op

def staggered_magnetization_sqr_op(hilbert, width, token_size : int = 1, z2_project = False):
    mag_func = lambda s, w : magnetization_staggered_sqr(s, w, token_size=token_size)
    mat_elements = OperatorMatrixElements(mag_func, width)

    op = TurboTurboOperator(
        hilbert=hilbert,
        operator=mat_elements,
        width=width,
        fast_operator=True, 
        z2_project=z2_project,
    )

    return op




def expect_lanczos(
    vstate,
    op,
    ham_op,
    *,
    chunk_size: Optional[int] = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    sigma = jnp.asarray(vstate.samples)
    if sigma.ndim == 3:
        sigma = sigma.reshape(-1, sigma.shape[-1])
    elif sigma.ndim == 1:
        sigma = sigma[jnp.newaxis, :]
    elif sigma.ndim != 2:
        raise ValueError(f"Unsupported sample shape {sigma.shape}.")

    batch_size = sigma.shape[0]
    if batch_size == 0:
        raise ValueError("expect_lanczos requires at least one Monte Carlo sample.")

    model = vstate._model
    params = vstate.parameters
    try:
        has_params_key = "params" in params
    except TypeError:
        has_params_key = False
    apply_params = params if has_params_key else {"params": params}

    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer or None.")

    width = int(getattr(ham_op, "_width", getattr(ham_op.operator, "width", 0)))
    token_size = int(getattr(model, "token_size", 1))
    ham_operator = ham_op.operator
    op_operator = op.operator

    # -------------------------------------------------------------------------
    # JAX-friendly, scan-based chunking for model.apply
    # -------------------------------------------------------------------------
    def _apply_full(configs: jnp.ndarray) -> jnp.ndarray:
        """Apply the model on a batch of configs, using lax.scan over chunks.

        This avoids a huge unrolled kernel when jitted, and limits the "live"
        batch size to `chunk_size` inside the loop.
        """
        n = configs.shape[0]

        if chunk_size is None or n <= chunk_size:
            return model.apply(apply_params, configs, mode="full")

        csize = int(chunk_size)
        n_chunks = (n + csize - 1) // csize
        pad = n_chunks * csize - n

        if pad > 0:
            pad_width = ((0, pad),) + ((0, 0),) * (configs.ndim - 1)
            configs_padded = jnp.pad(configs, pad_width)
        else:
            configs_padded = configs

        # (n_chunks, chunk_size, *sites_shape)
        configs_chunks = configs_padded.reshape(
            (n_chunks, csize) + configs.shape[1:]
        )

        def scan_body(carry, chunk):
            # chunk: (chunk_size, N_sites)
            out_chunk = model.apply(apply_params, chunk, mode="full")
            return carry, out_chunk

        _, out_chunks = jax.lax.scan(scan_body, None, configs_chunks)
        # out_chunks: (n_chunks, chunk_size, *out_shape)

        out_padded = out_chunks.reshape(
            (n_chunks * csize,) + out_chunks.shape[2:]
        )
        return out_padded[:n]

    # log ψ(σ)
    log_psi_sigma = _apply_full(sigma)

    # -------------------------------------------------------------------------
    # Helper to accumulate local contributions
    # -------------------------------------------------------------------------
    def _accumulate_local_contributions(
        lengths: jnp.ndarray,
        log_psi_conn: jnp.ndarray,
        mels: jnp.ndarray,
    ) -> jnp.ndarray:
        sections = jnp.cumsum(lengths)
        idx = get_idx(sections)
        delta = log_psi_conn - log_psi_sigma[idx]
        contribs = mels * jnp.exp(delta)
        return jnp.zeros((batch_size,), dtype=log_psi_conn.dtype).at[idx].add(contribs)

    # -------------------------------------------------------------------------
    # H |ψ> local estimator
    # -------------------------------------------------------------------------
    eta_win, mels, lengths, sites = ham_operator.apply_operator(sigma)
    lengths = jnp.asarray(lengths, dtype=jnp.int32)
    mels = jnp.asarray(mels)
    has_eta = eta_win.shape[0] > 0
    if has_eta:
        eta_full = _embed_windows_into_full(
            sigma,
            jnp.asarray(eta_win),
            lengths,
            jnp.asarray(sites),
            width=width,
            token_size=token_size,
        )

    # -------------------------------------------------------------------------
    # H² |ψ> via apply_operator_power
    # -------------------------------------------------------------------------
    eta_sq, mels_sq, lengths_sq = apply_operator_power(
        ham_operator,
        sigma,
        power=2,
        width=width,
        token_size=token_size,
    )
    lengths_sq = jnp.asarray(lengths_sq, dtype=jnp.int32)
    mels_sq = jnp.asarray(mels_sq)
    has_eta_sq = eta_sq.shape[0] > 0

    # -------------------------------------------------------------------------
    # Reuse model evaluations for H|ψ> and H²|ψ>
    # -------------------------------------------------------------------------
    if has_eta and has_eta_sq:
        n_eta = eta_full.shape[0]
        n_eta_sq = eta_sq.shape[0]
        all_confs = jnp.concatenate([eta_full, eta_sq], axis=0)
        log_psi_all = _apply_full(all_confs)
        log_psi_eta = log_psi_all[:n_eta]
        log_psi_eta_sq = log_psi_all[n_eta:]
        E_loc = _accumulate_local_contributions(lengths, log_psi_eta, mels)
        E_loc2 = _accumulate_local_contributions(lengths_sq, log_psi_eta_sq, mels_sq)
    elif has_eta:
        log_psi_eta = _apply_full(eta_full)
        E_loc = _accumulate_local_contributions(lengths, log_psi_eta, mels)
        E_loc2 = jnp.zeros((batch_size,), dtype=log_psi_sigma.dtype)
    elif has_eta_sq:
        log_psi_eta_sq = _apply_full(eta_sq)
        E_loc = jnp.zeros((batch_size,), dtype=log_psi_sigma.dtype)
        E_loc2 = _accumulate_local_contributions(lengths_sq, log_psi_eta_sq, mels_sq)
    else:
        E_loc = jnp.zeros((batch_size,), dtype=log_psi_sigma.dtype)
        E_loc2 = jnp.zeros((batch_size,), dtype=log_psi_sigma.dtype)

    # Real-valued energies
    E_loc = jnp.real(E_loc)
    E_loc2 = jnp.real(E_loc2)

    # -------------------------------------------------------------------------
    # Moments and Lanczos weights
    # -------------------------------------------------------------------------
    E = jnp.mean(E_loc)
    E2 = jnp.mean(E_loc**2)
    var = E2 - E**2
    v = jnp.sqrt(jnp.maximum(var, 0.0))
    E3 = jnp.mean(E_loc * E_loc2)

    v_safe = jnp.maximum(v, 1e-12)
    r = (E3 - 3.0 * E2 * E + 2.0 * E**3) / (2.0 * v_safe**3)
    alpha = r - jnp.sqrt(r**2 + 1.0)
    beta = alpha / (v_safe - alpha * E)

    delta_E_loc = E_loc - E 
    weights = 1 / (1+ alpha**2) * (1 + alpha/v_safe * delta_E_loc)**2 

    # -------------------------------------------------------------------------
    # Diagonal observable: one contribution per sample
    # -------------------------------------------------------------------------
    _, mels_diag, lengths_diag, _ = op_operator.apply_operator(sigma)
    lengths_diag = jnp.asarray(lengths_diag, dtype=jnp.int32)
    if not bool(jnp.all(lengths_diag == 1)):
        raise ValueError("Diagonal operator must contribute exactly once per sample.")

    mels_diag = jnp.asarray(mels_diag)
    if mels_diag.ndim != 1:
        mels_diag = mels_diag.reshape(-1,)
    if mels_diag.shape[0] != batch_size:
        raise ValueError(
            f"Diagonal operator mismatch: got {mels_diag.shape[0]} contributions "
            f"for {batch_size} samples."
        )
    O_vals = mels_diag

    # -------------------------------------------------------------------------
    # Weighted expectation
    # -------------------------------------------------------------------------
    O_lanczos = jnp.mean(weights * O_vals)

    return jnp.asarray(O_lanczos), jnp.asarray(weights)
