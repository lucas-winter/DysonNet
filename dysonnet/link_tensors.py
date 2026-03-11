import jax 
import jax.numpy as jnp
from functools import partial 
from dysonnet.partial_evaluation import unwrap
from dysonnet.S4 import get_kernel_params, green_roots



def get_complexify_matrix(k, dtype=jnp.complex64):
    dtype = jnp.result_type(dtype, 1j)
    N = k // 2
    eye = jnp.eye(N, dtype=dtype)
    M = jnp.concatenate([eye, 1j * eye], axis=1)     # shape (N, 2*N)
    return M                  

def get_decomplexify_matrix(d: int, dtype=jnp.complex64) -> jnp.ndarray:
    """
    Return M ∈ C^{2n×n} so that for any y ∈ C^n,
        (M @ y).real == [Re(y0),...,Re(y_{n-1}), Im(y0),...,Im(y_{n-1})].
    """
    d = d // 2
    dtype = jnp.result_type(dtype, 1j)
    I = jnp.eye(d, dtype=dtype)
    # Stack I (to pick out y) and -i·I (so that real(-i·y) = imag(y))
    M = jnp.vstack([ I,
                     -1j * I ])
    return M


def get_green_function_from_layer(params, layer_index, l_max, bidirectional):

    s4_params = params[f"blocks_{layer_index}"]["mixer"]["s4_kernel"]
    Lambda, P, Q, B, C, delta = get_kernel_params(s4_params)

    @jax.jit 
    def get_green_roots(Lambda, P, Q, B, C, delta):
        G_roots_complex = green_roots(
                Lambda, P, P, B, C, delta, l_max
            ) 
        
        current_G_roots = G_roots_complex[:l_max]
        if bidirectional: 
            current_G_roots = current_G_roots + jnp.roll(current_G_roots[::-1], 1)
        return current_G_roots

    g_func = jax.vmap(get_green_roots, in_axes=(-1, -1,-1,-1,-1, -1), out_axes=-1)(Lambda, P, Q, B, C, delta)
    return g_func 



def get_green_functions(params, cache, block_index =0, second_cached : bool = True, l_max:int = 1024, bidirectional: bool = True):
    # Get the data tensors 

    # Get the greens functions 
    groots1 = cache[f"blocks_{block_index}"]["mixer"]["s4_kernel"]["G_roots"]
    groots2 = cache[f"blocks_{block_index+1}"]["mixer"]["s4_kernel"]["G_roots"]

    # if second_cached:
    #     groots2 = cache[f"blocks_{block_index+1}"]["mixer"]["s4_kernel"]["G_roots"]
    # else: 
    #     groots2 = get_green_function_from_layer(params, block_index + 1, l_max, bidirectional)

    L = groots1.shape[0]  # Length of the sequence
    groots_1_rs = jnp.fft.ifft(groots1, axis=0, n = L)  # (B, L, d)
    groots_2_rs = jnp.fft.ifft(groots2, axis=0, n = L)  # (B, L, d)

    ssmD1 = params[f"blocks_{block_index}"]["mixer"]["s4_kernel"]["D"]
    ssmD2 = params[f"blocks_{block_index+1}"]["mixer"]["s4_kernel"]["D"]

    two = jnp.asarray(2, dtype=groots_1_rs.dtype)
    groots_1_rs = groots_1_rs.at[0].add(two*ssmD1[0])
    groots_2_rs = groots_2_rs.at[0].add(two*ssmD2[0]) 

    return groots_1_rs, groots_2_rs


def get_D_from_params(params, intermediates, block_index = 0):
    out_proj_base = params[f"blocks_{block_index}"]["mixer"]["out_proj"]["kernel"]
    d = out_proj_base.shape[0]  # d is the state dimension
    in_proj_base = params[f"blocks_{block_index+1}"]["mixer"]["in_proj_x"]["kernel"]
    act_gate = unwrap(intermediates[f"blocks_{block_index}"]["mixer"]["activated_gate"])
    #jac_ln = params["blocks_1"]["ln_jac_x"][0]
    ln_scale = params[f"blocks_{block_index+1}"]["LayerNorm_x"]["scale"]
    # Get the complexify and decomplexify matrices
    M_complexify = get_complexify_matrix(d, dtype=out_proj_base.dtype)
    M_decomplexify = get_decomplexify_matrix(d, dtype=out_proj_base.dtype)

    # Compute D
    D = M_decomplexify 
    D = jnp.einsum('ij, jk->ik', out_proj_base.T, D)  # (2d, d)
    D = jnp.einsum('ij, bli -> blij', D, act_gate)
    #D = jnp.einsum('blij, bljk-> blik', jac_ln, D)  # (B, L, d, 2d) # ATTENTION : Temporarily deactivate layer norm for debugging
    D = jnp.einsum('blij, i -> blij', D, ln_scale)  # (B, L, d, 2d)
    D = jnp.einsum('blij, ik->blkj', D, in_proj_base)  # (B, L, d, 2d)
    D_old = D
    D = jnp.einsum('blij, ik->blkj', D_old, M_complexify.T)
    Dconj = jnp.einsum('blij, ik->blkj', D_old.conj(), M_complexify.T)

    return D, Dconj

# ——— JITTED FAST CHAIN ———
@jax.jit
def _compute_D_fast(out_proj_base,
                    in_proj_base,
                    act_gate,
                    ln_scale,
                    M_complexify,
                    M_decomplexify):
    # same first step
    D0 = out_proj_base.T@M_decomplexify
    Dbase = act_gate[:, :, :, None] * D0[None, None, :, :]

    # fuse ln_scale into in_proj up front
    in_proj_fused = ln_scale[:, None] * in_proj_base  # shape (d, 2d)
    #in_proj_complex  = jnp.einsum('ji, ik->kj', in_proj_fused, M_complexify.T)
    in_proj_complex = (in_proj_fused@M_complexify.T).T # shape (2d, d)
    # one broadcast matmul for D and conj(D)
    
    Dstack = jnp.stack([Dbase, jnp.conj(Dbase)], axis=0)

    # Contract the I-axis of Dstack (axis 3) with the I-axis of in_proj_complex (axis 1):
    #   in_proj_complex: (K, I)
    #   R:                (2, B, L, J, K)
    R = jnp.tensordot(Dstack,
                     in_proj_complex,
                     axes=([3], [1]))

    # Unpack and do a single transpose per output:
    #   R[0]: (B, L, J, K) → .transpose(0,1,3,2) → (B, L, K, J)
    #   R[1]: same for the conjugate
    D     = R[0].transpose(0, 1, 3, 2)
    Dconj = R[1].transpose(0, 1, 3, 2)

    return D, Dconj


# ——— UPDATED BASE FUNCTIONS ———


def get_D_from_params_fast(params, intermediates, block_index=0):
    out_proj_base = params[f"blocks_{block_index}"]["mixer"]["out_proj"]["kernel"]
    d = out_proj_base.shape[0]  # d is the state dimension
    in_proj_base = params[f"blocks_{block_index+1}"]["mixer"]["in_proj_x"]["kernel"]
    act_gate = unwrap(intermediates[f"blocks_{block_index}"]["mixer"]["activated_gate"])
    #jac_ln = params["blocks_1"]["ln_jac_x"][0]
    ln_scale = params[f"blocks_{block_index+1}"]["LayerNorm_x"]["scale"]
    # Get the complexify and decomplexify matrices
    M_complexify = get_complexify_matrix(d, dtype=out_proj_base.dtype)
    M_decomplexify = get_decomplexify_matrix(d, dtype=out_proj_base.dtype)

    return _compute_D_fast(out_proj_base,
                           in_proj_base,
                           act_gate,
                           ln_scale,
                           M_complexify,
                           M_decomplexify)



@partial(jax.jit, static_argnames=('cutoff', 'width', 'symmetric'))
def two_layer_link_tensor(        
        D_fft : jnp.ndarray,
        groots1 : jnp.ndarray,
        groots2 : jnp.ndarray,
        deltaj : int, 
        width : int = -1,
        cutoff : int = -1, 
        symmetric: bool = True,): 
    
    if cutoff > 0:
        groots1 = groots1.at[cutoff:, :].set(0.0) 
        groots2 = groots2.at[cutoff:, : ].set(0.0) 

    if width > 0:
        groots1 = groots1.at[:width, :].set(0.0)
        groots1 = groots1.at[-width:, :].set(0.0)

    groots2 = jnp.roll(jnp.flip(groots2, axis=0), 1-deltaj, axis=0)  # (B, L)
    if symmetric:
        groots2 = (groots2 + jnp.roll(jnp.flip(groots2, axis=0), 1+deltaj, axis=0))/2

    D_fft = jnp.einsum('blji -> blji', D_fft)
    groots = jnp.einsum('li, lj -> lji', groots1, groots2)  # (B, L, d)
    groots_fft=  jnp.fft.fft(groots, axis=0, n = D_fft.shape[1])  # (B, L)
    D_real_space = jnp.fft.ifft(D_fft * groots_fft[jnp.newaxis, :], axis=1, n = D_fft.shape[1])  # (B, L, d) 

    return D_real_space

@partial(jax.jit, static_argnames=('symmetric'))
def compute_link_tensors(D, Dconj, groots_1_rs, groots_2_rs, deltaj_range, symmetric=False):
    """
    Computes the link tensors for two layers of S4.
    
    Args:
        D: The complexified projection matrix (B, L, d, 2d).
        groots_1_rs: The roots of the first layer (B, L, d).
        groots_2_rs: The roots of the second layer (B, L, d).
        deltaj_range: The range of shifts in the second layer.
        width: The width of the filter.
        cutoff: The cutoff for the roots.
        
    Returns:
        link_tensor: The link tensor (B, L, d, d).
    """

    complex_dtype = jnp.result_type(D.dtype, groots_1_rs.dtype, 1j)
    D_fft = jnp.fft.fft(D.astype(complex_dtype), axis=1, n = D.shape[1])  # (B, L) 
    D_conj_fft = jnp.fft.fft(Dconj.astype(complex_dtype), axis=1, n = D.shape[1])  # (B, L)

    link_tensor = jax.vmap(lambda deltaj: two_layer_link_tensor(
        D_fft, groots_1_rs, groots_2_rs, deltaj, symmetric=symmetric
    ))(deltaj_range)  # (B, L, d, d)
    link_tensor_conj = jax.vmap(lambda deltaj: two_layer_link_tensor(
        D_conj_fft, groots_1_rs.conj(), groots_2_rs, deltaj, symmetric=symmetric
    ))(deltaj_range)  # (B, L, d, d)
    return link_tensor, link_tensor_conj

@partial(jax.jit, static_argnames=('symmetric'))
def compute_link_tensors_real(D, Dconj, groots_1_rs, groots_2_rs, delta_j_range, symmetric=False):
    link_tensor, link_tensor_conj = compute_link_tensors(D, Dconj, groots_1_rs, groots_2_rs, delta_j_range, symmetric=symmetric)
    M_complexify = get_complexify_matrix(2*D.shape[-1], dtype=D.dtype)
    M_decomplexify = get_decomplexify_matrix(2*D.shape[-1], dtype=D.dtype)
    
    link_tensor = jnp.einsum('Δblij, jk->Δblik', link_tensor, M_complexify)
    link_tensor_conj = jnp.einsum('Δblij, jk->Δblik', link_tensor_conj, M_complexify.conj())

    link_tensor = (link_tensor + link_tensor_conj) / 2 # Real part (note this does not actually make it real though)
    
    link_tensor = jnp.einsum('Δblij, ki->Δblkj', link_tensor, M_decomplexify)  # (B, Δ, L, d)
    link_tensor = jnp.einsum('Δblkj -> bΔlkj', link_tensor)
    return link_tensor.real.astype(D.dtype)

@jax.jit
def link_multiply_real(link_tensor, x): 
    @jax.jit
    def roll(x, shift):
        """Rolls the array x by shift along the first axis."""
        return jnp.roll(x, shift, axis=1)

    #idx, _ = centers
    w = link_tensor.shape[1]  # Width of the filter
    shift = w % 2
    delta_j = jnp.arange(-w//2+shift, w//2+shift)
    y = jnp.einsum('bΔlij, blj -> bΔli', link_tensor, x)  
    y = jax.vmap(roll, in_axes=(1, 0), out_axes=1)(y, delta_j)
    y = jnp.sum(y, axis=1)

    return y 
