import jax 
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal
from typing import Any, Callable, Tuple, Optional
import jax.random as jr
from functools import partial


"""
This is an implementation of S4 inspired by the annotated S4 reference:
https://srush.github.io/annotated-s4/

Compared to the annotated S4 we use circuar convolution for periodic boundary conditions.
The S4 layers are implemented non-causal and can be made bidirectional.
Initializations are still done with the HiPPO matrix.
Weights are generally complex in this implementation.

Acknowledgement: Several functions below are adapted from the annotated S4
reference implementation and retain its mathematical structure.
"""

def discretize(A, B, C, delta): 
    """
    Discretize the matrices A, B, C using the bilinear method 
    \bar A = (I - \Delta/2 * A)^(-1) * (A * \Delta/2 + I)
    \bar B = (I - \Delta/2 * A)^(-1) * B * \Delta
    \bar C = C 

    return \bar A, \bar B, \bar C

    Acknowledgement: Based on the annotated S4 reference implementation:
    https://srush.github.io/annotated-s4/
    """
    I = jnp.eye(A.shape[0])
    BL = jnp.linalg.inv(I - A*(delta/2.0))
    Ab  =BL @ (A*(delta/2.0) + I)
    Bb  = (BL * delta) @ B 
    return Ab, Bb, C 


def circular_convolution(u, G_roots):
    """
    Computes circular convolution of u and K using FFT.
    Assumes u and K are real and have the same length L.
    """
    L = u.shape[0]
    assert G_roots.shape[0] == L, f"Input length {L} must match kernel length {K.shape[0]}"

    # Compute real FFTs of u and K
    ud = jnp.fft.rfft(u)
    #Kd = jnp.fft.rfft(K) # K is real as computed by kernel_DPLR in S4Layer

    # Element-wise multiplication in the frequency domain
    out_d = ud * G_roots

    # Inverse real FFT to get the circular convolution result
    # Specify the output length L explicitly
    y = jnp.fft.irfft(out_d, n=L)

    return y

def circular_convolution_complex(u, G_roots):
    """
    Computes circular convolution of u and K using FFT.
    Assumes u and K are real and have the same length L.
    """
    L = u.shape[0]
    assert G_roots.shape[0] == L, f"Input length {L} must match kernel length {K.shape[0]}"

    # Compute real FFTs of u and K
    ud = jnp.fft.fft(u)

    # Element-wise multiplication in the frequency domain
    out_d = ud * G_roots

    # Inverse real FFT to get the circular convolution result
    # Specify the output length L explicitly
    y = jnp.fft.ifft(out_d, n=L)

    return y

## Hippo matrix 
def hippo_matrix(N):
    """
    Generate a Hippo matrix of size n x n

    Acknowledgement: Based on the annotated S4 reference implementation:
    https://srush.github.io/annotated-s4/
    """

    P = jnp.sqrt(2*jnp.arange(0, N) + 1)
    A = P[:, jnp.newaxis] * P[jnp.newaxis, :]
    return -(jnp.tril(A) - jnp.diag(jnp.arange(N)))
    
def make_NLPR_HiPPO(N):
    """
    Generate the HiPPO matrix in a format of a normal matrix + low rank PQ correction 

    Input:
    N: size of the HiPPO matrix

    Output:
    A: normal matrix    
    P, Q: low rank corrections

    Acknowledgement: Based on the annotated S4 reference implementation:
    https://srush.github.io/annotated-s4/
    """

    nhippo = hippo_matrix(N)

    P = jnp.sqrt(jnp.arange(N) + 0.5) 
    B = jnp.sqrt(2 * jnp.arange(N) + 1.0)
    return nhippo, P, B

def make_DPLR_HiPPO(N, dtype=jnp.complex64):
    """
    Diagonalize NPLR representation.

    Acknowledgement: Based on the annotated S4 reference implementation:
    https://srush.github.io/annotated-s4/
    """
    A, P, B = make_NLPR_HiPPO(N)

    S = A + P[:, jnp.newaxis] * P[jnp.newaxis, :]

    # Check skew symmetry
    S_diag = jnp.diagonal(S)
    Lambda_real = jnp.mean(S_diag) * jnp.ones_like(S_diag)
    # assert np.allclose(Lambda_real, S_diag, atol=1e-3)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = jnp.linalg.eigh(S * -1j)

    P = V.conj().T @ P
    B = V.conj().T @ B
    dtype = jnp.dtype(dtype)
    real_dtype = jnp.result_type(dtype, jnp.float32)
    Lambda = Lambda_real.astype(real_dtype) + 1j * Lambda_imag.astype(real_dtype)
    Lambda = Lambda.astype(dtype)
    return Lambda, P.astype(dtype), B.astype(dtype), V.astype(dtype)



def make_random_NPLR(
    key: jax.random.PRNGKey,
    N: int,
    rank: int = 1,
    decay_scale: float = 1.0,
    freq_scale: float = 1.0,
    long_range_frac: float = 0.5,
    dtype: jnp.dtype = jnp.complex64
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generates a random Normal + Low Rank (NPLR) matrix S = A + PQ^H
    with a balance of decaying and long-range modes in the normal part A.

    Args:
        key: JAX random key.
        N: Size of the matrix (NxN).
        rank: Rank of the low-rank correction PQ^H (default: 1).
        decay_scale: Controls the magnitude of negative real parts of eigenvalues.
        freq_scale: Controls the magnitude of imaginary parts of eigenvalues.
        long_range_frac: Fraction of eigenvalues intended to be 'long-range'.
        dtype: Data type for the matrices (default: jnp.complex64).

    Returns:
        A: Normal matrix (N, N).
        P: Low-rank factor (N, rank).
        Q: Low-rank factor (N, rank).
    """
    assert 0.0 <= long_range_frac <= 1.0, "long_range_frac must be between 0 and 1"
    real_dtype = jnp.result_type(dtype, jnp.float32)
    complex_dtype = jnp.result_type(real_dtype, 1j)
    if N == 0:
        return jnp.zeros((0, 0), dtype=complex_dtype), jnp.zeros((0, rank), dtype=complex_dtype), jnp.zeros((0, rank), dtype=complex_dtype)

    key_eig_re, key_eig_im, key_V, key_P, key_Q, key_perm = jr.split(key, 6)

    # 1. Generate Eigenvalues (Lambda diagonal)
    N_long = int(N * long_range_frac)
    N_decay = N - N_long

    re_long = -jr.uniform(key_eig_re, shape=(N_long,), minval=0.0, maxval=0.1 * decay_scale, dtype=real_dtype)
    min_decay = jnp.minimum(-0.1 * decay_scale - 1e-6, -decay_scale)
    max_decay = -0.1 * decay_scale
    re_decay = jr.uniform(key_eig_re, shape=(N_decay,), minval=min_decay, maxval=max_decay, dtype=real_dtype)
    re_parts = jnp.concatenate([re_long, re_decay])
    im_parts = jr.normal(key_eig_im, shape=(N,), dtype=real_dtype) * freq_scale
    Lambda_diag = (re_parts + 1j * im_parts).astype(complex_dtype)
    Lambda_diag = jr.permutation(key_perm, Lambda_diag)

    # 2. Generate Random Unitary Matrix (V)
    key_V_real, key_V_imag = jr.split(key_V)
    Z_real = jr.normal(key_V_real, shape=(N, N), dtype=real_dtype)
    Z_imag = jr.normal(key_V_imag, shape=(N, N), dtype=real_dtype)
    Z = Z_real + 1j * Z_imag
    V, _ = jnp.linalg.qr(Z)

    # 3. Construct Normal Matrix A = V Lambda V^H
    A = V @ jnp.diag(Lambda_diag) @ V.conj().T

    # 4. Generate Low-Rank Factors (P, Q)
    key_P_real, key_P_imag = jr.split(key_P)
    P_real = jr.normal(key_P_real, shape=(N, rank), dtype=real_dtype)
    P_imag = jr.normal(key_P_imag, shape=(N, rank), dtype=real_dtype)
    P = P_real + 1j * P_imag

    key_Q_real, key_Q_imag = jr.split(key_Q)
    Q_real = jr.normal(key_Q_real, shape=(N, rank), dtype=real_dtype)
    Q_imag = jr.normal(key_Q_imag, shape=(N, rank), dtype=real_dtype)
    Q = Q_real + 1j * Q_imag

    A = A.astype(complex_dtype)
    P = P.astype(complex_dtype)
    Q = Q.astype(complex_dtype)
    
    # Handle rank=0 case P, Q generation
    if rank == 0:
        P = jnp.zeros((N, 0), dtype=complex_dtype)
        Q = jnp.zeros((N, 0), dtype=complex_dtype)

    return A, P, Q


def make_DPLR_random(
    key: jax.random.PRNGKey,
    N: int,
    decay_scale: float = 1.0,
    freq_scale: float = 1.0,
    long_range_frac: float = 0.5,
    dtype: jnp.dtype = jnp.complex64
) :
    """
    Generates a random Normal + Low Rank (NPLR) matrix S = A + PQ^H
    and diagonalizes the full matrix S using general eigenvalue decomposition
    such that S = V Lambda V^-1.

    Returns the diagonal representation (Lambda, P_diag, Q_diag, V),
    where Lambda contains the eigenvalues of S, P_diag = V^-1 P,
    Q_diag = V^H Q, and V contains the right eigenvectors of S.

    Args:
        key: JAX random key.
        N: Size of the matrix (NxN).
        rank: Rank of the low-rank correction PQ^H.
        decay_scale: Controls the magnitude of negative real parts of A's eigenvalues.
        freq_scale: Controls the magnitude of imaginary parts of A's eigenvalues.
        long_range_frac: Fraction of A's eigenvalues intended to be 'long-range'.
        dtype: Data type for the matrices (default: jnp.complex64).

    Returns:
        Lambda: Eigenvalues of S (N,). Complex valued in general.
        P_diag: Transformed P factor (N, rank). P_diag = V^-1 @ P.
        Q_diag: Transformed Q factor (N, rank). Q_diag = V^H @ Q.
        V: Right eigenvector matrix of S (N, N). S @ V = V @ diag(Lambda).
    """
    rank = 1 # Default rank for NPLR generation

    # 1. Generate the random NPLR components (Normal A, Low-Rank P, Q)
    A, P, Q = make_random_NPLR(
        key, N, rank, decay_scale, freq_scale, long_range_frac, dtype
    )

    # Handle N=0 edge case immediately after generation
    if N == 0:
        Lambda_empty = jnp.zeros((0,), dtype=dtype)
        # P, Q are already (0, rank) from make_random_NPLR
        V_empty = jnp.zeros((0, 0), dtype=dtype)
        return Lambda_empty, P, Q, V_empty # P, Q act as P_diag, Q_diag here

    # 2. Construct the full matrix S = A + P Q^H
    # Handle rank=0 case where P@Q^H is zero matrix of shape (N,N)
    if rank > 0:
        S = A + P @ Q.conj().T
    else:
        # If rank is 0, P/Q have shape (N, 0), P@Q.conj().T is zeros(N,N)
        S = A # Effectively S = A + zeros(N,N)

    # 3. Diagonalize S using general eigenvalue decomposition S = V Lambda V^-1
    # S is generally not Hermitian, so use jnp.linalg.eig
    Lambda, V = jnp.linalg.eig(S)

    # 4. Calculate V inverse needed for transforming P
    # Add error handling for potentially singular V (though unlikely for random S)
    try:
        V_inv = jnp.linalg.inv(V)
    except jnp.linalg.LinAlgError:
        # Fallback to pseudo-inverse if V is singular or ill-conditioned
        print(f"Warning: Eigenvector matrix V for N={N}, rank={rank} is singular or near-singular. Using pseudo-inverse.")
        V_inv = jnp.linalg.pinv(V)

    # 5. Transform P and Q factors based on the change of basis z = V^-1 x
    # P_diag = V^-1 @ P
    # Q_diag = V^H @ Q  (Derived from C' = C @ V for output matrix C=Q^H)
    if rank > 0:
        P_diag = V_inv @ P
        Q_diag = V.conj().T @ Q
    else: # rank == 0
        # P, Q have shape (N, 0), so P_diag, Q_diag should also be (N, 0)
        P_diag = jnp.zeros((N, 0), dtype=dtype)
        Q_diag = jnp.zeros((N, 0), dtype=dtype)


    # Ensure consistent dtypes for return values, matching input dtype
    Lambda = Lambda.astype(dtype)
    P_diag = P_diag.astype(dtype)
    Q_diag = Q_diag.astype(dtype)
    V = V.astype(dtype) # V from eig should already be complex if dtype is complex

    # Ensure correct shapes 
    if rank == 1: 
        P_diag = P_diag[:, 0]
        Q_diag = Q_diag[:, 0]

    return Lambda, P_diag, Q_diag, V

    

# Factory for constant initializer in Flax
def init(x):
    def _init(key, shape):
        #assert shape == x.shape
        #return x
        return jnp.broadcast_to(x, shape)

    return _init

def hippo_initializer(N, dtype=jnp.complex64):
    Lambda, P, B, _ = make_DPLR_HiPPO(N, dtype=dtype)
    real_dtype = jnp.result_type(dtype, jnp.float32)
    return init(Lambda.real.astype(real_dtype)), init(Lambda.imag.astype(real_dtype)), init(P.astype(dtype)), init(B.astype(dtype))

def random_initializer(N, decay_scale=1.0, freq_scale=1.0, long_range_frac=0.5, seed = 42, dtype=jnp.complex64):
    """Initialize S4 parameters using random DPLR method."""
    key = jr.PRNGKey(seed)  # Fixed seed for deterministic initialization
    Lambda, P, Q, _ = make_DPLR_random(key, N, decay_scale, freq_scale, long_range_frac, dtype=dtype)
    # Extract real and imaginary parts of Lambda
    Lambda_re = Lambda.real
    Lambda_im = Lambda.imag
    # Use Q for B parameter
    B = Q  
    real_dtype = jnp.result_type(dtype, jnp.float32)

    return init(Lambda_re.astype(real_dtype)), init(Lambda_im.astype(real_dtype)), init(P.astype(dtype)), init(B.astype(dtype))

# Convolution kernel 
def convolution_kernel_from_greens_function(green_fun, L):
    """
    Compute the convolution kernel from the Green's function at the roots of unity 

    Input:
    green_fun: function that computes the Green's function at the roots of unity
    L: length of the kernel 

    Output:
    kernel: convolution kernel of length L
    """

    roots=  jnp.exp(1j*2*jnp.pi*jnp.arange(L)/L) # roots of unity
    atRoots = jax.vmap(green_fun)(roots)
    out = jnp.fft.ifft(atRoots, L).reshape(L)
    return out.real
    
@jax.jit
def cauchy(v, omega, lambd):
    """
    Cauchy matrix multiplication: (n), (l), (n) -> (l).

    Acknowledgement: Based on the annotated S4 reference implementation:
    https://srush.github.io/annotated-s4/
    """
    cauchy_dot = lambda _omega: (v / (_omega - lambd)).sum()
    return jax.vmap(cauchy_dot)(omega)



def green_DPLR(Lambda, P, Q, B, C, delta, L): 
    """
    Compute the convolution kernel K from the DPLR representation of the S4 layer. Note this implements

    K(z) = c(z) g_{z, Lambda}(C,B) - g_{z, Lambda}(C, P) (1 + k_{z, Lmabda}(q^* P))^(-1) k_{z, Lambda}(Q*, B)

    where g_{z, Lambda}(A, B) = (I - z A)^(-1) B

    Input:
    Lambda: Diagonal elements of the A matrix
    P, Q: Low rank corrections
    B, C: B, C matrices
    delta: time step
    unmat: if True return the unmatriced version of the kernel

    Output:
    K: convolution kernel of the S4 layer

    Acknowledgement: Based on the annotated S4 reference implementation:
    https://srush.github.io/annotated-s4/
    """
    Omega_L = jnp.exp((-2j * jnp.pi) * (jnp.arange(L) / L))

    aterm = (C.conj(), Q.conj())
    bterm = (B, P)

    g = (2.0 / delta) * ((1.0 - Omega_L) / (1.0 + Omega_L))
    c = 2.0 / (1.0 + Omega_L)

    # Reduction to core Cauchy kernel
    k00 = cauchy(aterm[0] * bterm[0], g, Lambda)
    k01 = cauchy(aterm[0] * bterm[1], g, Lambda)
    k10 = cauchy(aterm[1] * bterm[0], g, Lambda)
    k11 = cauchy(aterm[1] * bterm[1], g, Lambda)
    atRoots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    out = jnp.fft.ifft(atRoots, L).reshape(L)
    return out.real



def green_roots(Lambda, P, Q, B, C, delta, L): 
    """
    Compute the convolution kernel K from the DPLR representation of the S4 layer. Note this implements

    K(z) = c(z) g_{z, Lambda}(C,B) - g_{z, Lambda}(C, P) (1 + k_{z, Lmabda}(q^* P))^(-1) k_{z, Lambda}(Q*, B)

    where g_{z, Lambda}(A, B) = (I - z A)^(-1) B

    Input:
    Lambda: Diagonal elements of the A matrix
    P, Q: Low rank corrections
    B, C: B, C matrices
    delta: time step
    unmat: if True return the unmatriced version of the kernel

    Output:
    K: convolution kernel of the S4 layer

    Acknowledgement: Based on the annotated S4 reference implementation:
    https://srush.github.io/annotated-s4/
    """
    Omega_L = jnp.exp((-2j * jnp.pi) * (jnp.arange(L) / L))

    aterm = (C.conj(), Q.conj())
    bterm = (B, P)

    g = (2.0 / delta) * ((1.0 - Omega_L) / (1.0 + Omega_L))
    c = 2.0 / (1.0 + Omega_L)

    # Reduction to core Cauchy kernel
    k00 = cauchy(aterm[0] * bterm[0], g, Lambda)
    k01 = cauchy(aterm[0] * bterm[1], g, Lambda)
    k10 = cauchy(aterm[1] * bterm[0], g, Lambda)
    k11 = cauchy(aterm[1] * bterm[1], g, Lambda)
    atRoots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    return atRoots



def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """
    Log-uniform initializer for S4 step sizes.

    Acknowledgement: Based on the annotated S4 reference implementation:
    https://srush.github.io/annotated-s4/
    """
    def init(key, shape):
        dtype = jnp.result_type(dt_min, dt_max, jnp.float32)
        return (jax.random.uniform(key, shape, dtype=dtype) * (
            jnp.log(dt_max) - jnp.log(dt_min)
        ) + jnp.log(dt_min)).astype(dtype)

    return init

def cloneLayer(layer):
    return nn.vmap(
        layer,
        in_axes=(-1, None, None),
        out_axes=-1,
        variable_axes={"params": -1, "cache": -1, "prime": -1}, #Prev version 
        split_rngs={"params": True},
    )

def get_pole_1d(lamb, P, Q, step):
    """
    Calculate the z-domain pole from S4 parameters.
    
    Args:
        lamb: Lambda eigenvalue (diagonal of A)
        P: P vector (from low-rank structure)
        Q: Q vector (from low-rank structure)
        step: Discretization step size (delta)
        
    Returns:
        z-domain pole
    """
    zp = (2 - step*(lamb - jnp.conj(Q) * P)) / (2 + step * (lamb - jnp.conj(Q) * P)) 
    return zp 

def map_to_splane(pole, step):
    """
    Map a z-domain pole to the s-domain (continuous time).
    
    Args:
        pole: z-domain pole
        step: Discretization step size
        
    Returns:
        s-domain pole
    """
    return 2/step * (pole - 1) / (pole + 1)


def discrete_DPLR(Lambda, P, Q, B, C, step, L):
    """
    Discretize DPLR parameters to (Abar, Bbar, Cbar).

    Acknowledgement: Based on the annotated S4 reference implementation:
    https://srush.github.io/annotated-s4/
    """
    # Convert parameters to matrices
    B = B[:, jnp.newaxis]
    Ct = C[jnp.newaxis, :]

    N = Lambda.shape[0]
    A = jnp.diag(Lambda) - P[:, jnp.newaxis] @ Q[:, jnp.newaxis].conj().T
    I = jnp.eye(N)

    # Forward Euler
    A0 = (2.0 / step) * I + A

    # Backward Euler
    D = jnp.diag(1.0 / ((2.0 / step) - Lambda))
    Qc = Q.conj().T.reshape(1, -1)
    P2 = P.reshape(-1, 1)
    A1 = D - (D @ P2 * (1.0 / (1 + (Qc @ D @ P2))) * Qc @ D)

    # A bar and B bar
    Ab = A1 @ A0
    Bb = 2 * A1 @ B

    # Recover Cbar from Ct
    Cb = C[jnp.newaxis, :].conj()
    
    # Cast correctly
    Ab = Ab.astype(jnp.complex64)
    Bb = Bb.astype(jnp.complex64)
    Cb = Cb.astype(jnp.complex64)

    return Ab, Bb, Cb


@jax.jit 
def affine_op(x, y):
    a1, b1 = x
    a2, b2 = y

    anew = a1@a2 
    bnew = a2@b1 + b2

    return (anew, bnew)


@jax.jit
def scan_SSM(Ab, Bb, Cb, u):
    Bu = (Bb@u)
    Bu = jnp.einsum('dl->ld', Bu)[:, :, jnp.newaxis]

    # Expand A to have a time axis matching Bx’s sequence length
    Ab = jnp.broadcast_to(Ab[None, :, :], (Bu.shape[0], Ab.shape[0], Ab.shape[1]))
    _, y = jax.lax.associative_scan(affine_op, (Ab, Bu), axis = 0)
    return Cb @ y

@jax.jit
def scan_SSM_periodic(Ab, AL_prod, AdeltaL, Bb, Cb, u):
    d = Ab.shape[0]

    Bu = (Bb@u)
    Bu = jnp.einsum('dl->ld', Bu)[:, :, jnp.newaxis]

    # Expand A to have a time axis matching Bx’s sequence length
    Ab = jnp.broadcast_to(Ab[None, :, :], (Bu.shape[0], Ab.shape[0], Ab.shape[1]))
    A_cum, y_part = jax.lax.associative_scan(affine_op, (Ab, Bu), axis = 0) # Note A has shape (L, B, d, d) while B has shape (L, B, d, 1)

    M = jnp.eye(d, dtype=AL_prod.dtype) - AL_prod
    yL = AdeltaL@y_part[-1]                   # (B,d,1)
    y0 = jnp.linalg.solve(M, yL)[..., 0]         # (B,d)

    
    # propagate y0 through time
    y_full = y_part + A_cum@y0[..., jnp.newaxis]  # (L,B,d)
    y = Cb@y_full                # (L,B,d_out)
    return y


def get_kernel_params(s4_params):
    B = s4_params["B_re"] + 1j * s4_params["B_im"]
    C = s4_params["C_re"] + 1j * s4_params["C_im"]
    Lambda = - jnp.abs(s4_params["Lambda_re"]) + 1j * s4_params["Lambda_im"]
    P = s4_params["P_re"] + 1j * s4_params["P_im"]
    delta = jnp.exp(s4_params["log_step"])

    return Lambda, P, P, B, C, delta # ATTENTION: removed conjugate here 


def get_tails(Lambda, P, Q, B, C, delta, lmax, bidirectional): 
    """
    Get the tails of the S4 kernel in both z-domain and s-domain.
    
    Returns:
        A tuple (z_tails, s_tails) where:
        - z_tails: Complex tails in z-domain (discrete time)
        - s_tails: Complex tails in s-domain (continuous time)
    """
    # Construct actual Lambda values (complex)
    g_roots = green_roots(
        Lambda, P, Q, B, C, delta, lmax
    )

    # Compute the response in real space
    real_space_G = jnp.fft.ifft(g_roots, n= lmax)
    real_space_G = real_space_G + jnp.roll(jnp.flip(real_space_G), 1)
    
    # Calculate the tail
    # Calculate the maximum possible Lcut value
    max_lcut = lmax // 2

    # Calculate the tail for each Lcut from 0 to max_lcut using a list comprehension
    # The slice real_space_G[Lcut:-Lcut] (or real_space_G[:] for Lcut=0) selects the central part.
    # jnp.mean handles empty slices (when Lcut = l_max // 2 and l_max is even) by returning NaN.
    tails = jnp.array([
        jnp.mean(real_space_G[Lcut:-Lcut if Lcut > 0 else None])
        for Lcut in range(max_lcut + 1)
    ])
    
    # tails is now an array where tails[i] corresponds to the mean calculated with Lcut = i.
    return tails


def get_tails_exact(Lambda, P, Q, B, C, delta, L, Lcut):
    """
    Compute the tails of the Green's function response.
    
    Args:
      lambda, P, Q, B, C: DPLR parameters, each of shape (N,) complex.
      delta:    discretization step
      L:        length of the sequence (number of roots of unity).
      Lcut:     length of the tail to be considered.

    Returns:
      tail:     complex tail vector of shape (L,)
    """
    # Compute the Green's function roots
    g_roots = green_roots(Lambda, P, Q, B, C, delta, L)
    
    # Compute the response in real space
    real_space_G = jnp.fft.ifft(g_roots, n=L)
    real_space_G = real_space_G + jnp.roll(jnp.flip(real_space_G), 1)
    
    # Calculate the tail
    # Calculate the tail using cumulative sums for efficiency
    N_window = L - (2 * Lcut + 1)
    if N_window <= 0:
       return jnp.zeros(2 * Lcut, dtype=real_space_G.dtype)
      # Alternatively, raise an error:


    # Concatenate G with itself to handle circular shifts easily via slicing
    G_concat = jnp.concatenate([real_space_G, real_space_G])
    # Compute cumulative sum, padded with 0 at the start for easier indexing
    cumsum_G_concat_padded = jnp.pad(jnp.cumsum(G_concat, dtype=real_space_G.dtype), (1, 0))

    # Indices for j (shifts)
    j_values = jnp.arange(-Lcut, Lcut + 1) # Shape (2 * Lcut + 1,)


    # Indices for the padded cumsum array (start and end+1)
    start_indices_cumsum = L - j_values + Lcut + 1
    # End index for sum calculation (exclusive) in G_concat
    end_indices_cumsum = L - j_values + L - Lcut 

    # Compute sums using the padded cumulative sum array: sum[a:b] = cumsum[b] - cumsum[a]
    window_sums = cumsum_G_concat_padded[end_indices_cumsum] - cumsum_G_concat_padded[start_indices_cumsum]

    # Compute the mean by dividing by the window size
    tail = window_sums / N_window

    mean_response = jnp.sum(real_space_G, axis = 0)
    #tail = jnp.mean(real_space_G)
    return tail, mean_response



def conv_kernel_factor(kernel, Lcut):
    L_k = kernel.shape[0]
    center = L_k//2 - 1 
    distance = jnp.abs(jnp.arange(L_k) - center) 

    mult = 1 - distance / Lcut
    return jnp.sum(kernel * mult[:, jnp.newaxis,jnp.newaxis], axis=0)



# -------------------------- Utility functions  -----------------------------


def convert_to_complex(vec):
    shape = vec.shape 
    vec_out = vec.reshape(shape[:-1] + (2, -1))
    vec_out = vec_out[... , 0, :] + 1j * vec_out[... , 1, :]
    return vec_out 


def convert_to_real(vec):
    vec_out = jnp.stack([vec.real, vec.imag], axis=-1).astype(vec.real.dtype)
    vec_out = jnp.einsum("...dc -> ...cd", vec_out)
    shape = vec_out.shape[:-2] + (-1,)
    vec_out = vec_out.reshape(shape)
    return vec_out

def circulant_matrix_slice(G_roots, width, L):
    """
    Build a circulant matrix slice for the given roots and width.

    Parameters
    ----------
    G_roots : complex[ L ]  – length-L roots
    width   : int            – width of the slice (2*width+1)
    L       : int            – full sequence length

    Returns
    -------
    K_block : complex[ M, M ] – circulant matrix slice of size (2*width+1, 2*width+1)
    """
    # indices inside the window: 0 … 2*width
    idx = jnp.arange(2 * width + 1)

    G_roots_real = jnp.fft.ifft(G_roots)  # real-space kernel of the full system (length L)

    # Build the M×M sub-circulant block: K_full[(i-j) mod L]
    offsets = (idx[:, None] - idx[None, :]) % L          # (M, M)
    K_block = G_roots_real[offsets]                            # (M, M)
    return K_block

@partial(jax.jit, static_argnames=('l_max',))
def greens_function_power_law(power_alpha_raw, power_c, l_max):

        eps = 0.1                                   # lattice shift
        # distances r = eps, 1+eps, 2+eps, … up to l_max–1+eps
        r = jnp.arange(l_max) + eps                  # shape (L,)
        r = r[None, :]                               # → (1, L) for broadcasting

        # (modes, L)
        power_alpha = 0.5 + 6 * nn.sigmoid(power_alpha_raw)


        kernel = r ** (-power_alpha[:, None])  # (modes, L)

        # force the zero‐distance site to zero
        kernel = kernel.at[:, 0].set(0)
        
        # Normalize by the size so whole thing is scale invariant 
        zeta = jnp.sum(jnp.abs(kernel), axis=-1, keepdims=True)  # (modes, 1)
        kernel = kernel / (zeta + 1e-6)

        kernel = power_c[:, None] * kernel # Rescale after normalization
        
        kernel = jnp.sum(kernel, axis=0)  # sum over modes → (L,)
        
        # fixed-length real → complex FFT  (L  →  L/2+1)
        kernel_fft = jnp.fft.fft(kernel, l_max, axis=-1)
       
        return kernel_fft.astype(jnp.complex64)  # return complex kernel


# ---------------------------------------------------------------------------
class S4Kernel(nn.Module):
    """
    S4 kernel implementation.

    Acknowledgement: Core S4 kernel structure follows the annotated S4
    reference implementation:
    https://srush.github.io/annotated-s4/
    """
    d_state: int # State dimension N
    l_max: int # Max sequence length L
    l_width: Optional[int] = None
    complex_input: bool = False # Whether the input u is complex
    real_dtype: Any = jnp.float32
    complex_dtype: Any = None
    init_type: str = "hippo"  # Options: "hippo", "random"
    decay_scale: float = 1.0  # For random initialization
    freq_scale: float = 1.0  # For random initialization
    long_range_frac: float = 0.5  # For random initialization
    bidirectional : bool = False # Whether to use bidirectional processing
    use_circulant_slice : bool = False # Whether to use circulant matrix slices
    normalize_s4 : bool = False 
    use_power_law : bool = False 
    power_law_modes : int = 1 
    power_init_range : float = 2.0# Range for power law initialization
    include_short_range : bool = False


    def setup(self):
        real_dtype = jnp.dtype(self.real_dtype)
        complex_dtype = None if self.complex_dtype is None else jnp.dtype(self.complex_dtype)
        complex_dtype = jnp.result_type(real_dtype, 1j) if complex_dtype is None else jnp.result_type(real_dtype, complex_dtype)
        # Choose initializer based on init_type
        if self.init_type == "hippo":
            Lambda_re_init, Lambda_im_init, P_init, B_init = hippo_initializer(self.d_state, dtype=complex_dtype)
        elif self.init_type == "random":
            Lambda_re_init, Lambda_im_init, P_init, B_init = random_initializer(
                self.d_state, self.decay_scale, self.freq_scale, self.long_range_frac, dtype=complex_dtype
            )
        else:
            raise ValueError(f"Unknown init_type: {self.init_type}")


        # Lambda is complex: Lambda = -Lambda_re + 1j * Lambda_im
        self.Lambda_re = self.param('Lambda_re', Lambda_re_init, (self.d_state,))
        self.Lambda_im = self.param('Lambda_im', Lambda_im_init, (self.d_state,))
        # Ensure Lambda_re is non-positive
        self.Lambda = -jnp.abs(self.Lambda_re) + 1j * self.Lambda_im
        self.Lambda = jnp.asarray(self.Lambda, dtype=self.complex_dtype) # Ensure complex type

        # P, B, C are complex-valued in general for S4D
        # Initialize P, B from Hippo (real) but allow them to become complex
        P_init_val = P_init(None, (self.d_state,)) # Get initial values
        B_init_val = B_init(None, (self.d_state,))
        # Create complex parameters P, B initialized with real Hippo values
        self.P_re = self.param("P_re", init(P_init_val.real.astype(real_dtype)), (self.d_state,))
        self.P_im = self.param("P_im", init(P_init_val.imag.astype(real_dtype)), (self.d_state,)) # Init imag part to 0
        self.P = jnp.asarray(self.P_re + 1j * self.P_im, dtype=complex_dtype)

        self.B_re = self.param("B_re", init(B_init_val.real.astype(real_dtype)), (self.d_state,))
        self.B_im = self.param("B_im", init(B_init_val.imag.astype(real_dtype)), (self.d_state,))
        self.B = jnp.asarray(self.B_re + 1j * self.B_im, dtype=complex_dtype)

        # C is complex-valued, initialized randomly
        # Shape (d_state,) required by green_roots/cauchy
        # If output is real, C should be conjugate of B? Check literature. Assume general complex C.
        C_init_re = lambda key, shape: normal(stddev=0.5**0.5)(key, shape).astype(real_dtype)
        C_init_im = lambda key, shape: normal(stddev=0.5**0.5)(key, shape).astype(real_dtype)
        self.C_re = self.param("C_re", C_init_re, (self.d_state,))
        self.C_im = self.param("C_im", C_init_im, (self.d_state,))
        self.C = jnp.asarray(self.C_re + 1j * self.C_im, dtype=complex_dtype)

        # D is a scalar, potentially complex. Initialize to real 1.0
        # Make D learnable and real for now
        self.D = self.param("D", lambda key, shape: jnp.ones(shape, dtype=real_dtype), (1,)) # Real D for now


        if self.normalize_s4:
            self.scale_greens_function = self.param("scale_green", lambda key, shape: jnp.ones(shape, dtype=real_dtype), (1,)) 

        # Q = P conjugate? For HiPPO properties. Let's assume Q = P.conj()
        self.Q = jnp.conj(self.P)

        # Delta (time step) is learned
        self.log_step = jnp.asarray(self.param("log_step", log_step_initializer(), (1,)), dtype=real_dtype)
        # Ensure delta is positive
        self.delta = jnp.exp(self.log_step[0]) # Extract scalar

        # Note: Pass parameters explicitly to ensure correct values are used
        G_roots_complex = green_roots(
            self.Lambda, self.P, self.Q, self.B, self.C, self.delta, self.l_max
        )
        self.G_roots = jnp.asarray(G_roots_complex, dtype=self.complex_dtype) # Store the complex roots

        if self.is_mutable_collection('cache'):
            self.ssm = self.variable('cache', 'ssm', lambda: self._build_ssm())
            
            if self.use_circulant_slice:
                current_G_roots = G_roots_complex
                if self.bidirectional:
                    current_G_roots = current_G_roots + jnp.roll(current_G_roots[::-1], 1)
                
                # Initialize circulant slice if using it
                self.circulant_slice = self.variable(
                    'cache', 'circulant_slice', 
                    lambda: circulant_matrix_slice(current_G_roots, self.l_width, self.l_max)
                )
            
            # Debugging variables to store intermediate values
            self.g_roots_debug = self.variable('cache', 'G_roots', lambda: G_roots_complex)
            self.s4_internal_u_debug = self.variable('cache', 's4_internal_u', lambda: jnp.zeros((self.l_max,), dtype=self.real_dtype))
            self.s4_internal_y_debug = self.variable('cache', 's4_internal_y', lambda: jnp.zeros((self.l_max,), dtype=self.complex_dtype))
            self.s4_internal_y_debug_after = self.variable('cache', 's4_internal_y_after', lambda: jnp.zeros((self.l_max,), dtype=self.complex_dtype))


        # ------------------------------------------------------------------
        # >>>  OPTIONAL 1/r^α POWER–LAW TAIL  <<<
        # ------------------------------------------------------------------
        if self.use_power_law:
            self.power_alpha_raw = self.param(
                "power_alpha_raw",
                # initialise near the centre of the range → α≈2
                lambda k, s: jax.random.normal(k, s),
                (self.power_law_modes,),
            )

            self.power_c_re = self.param(
                "power_c_re",
                lambda k, s: self.power_init_range*jax.random.normal(k, s),
                (self.power_law_modes,),
            )
            # self.power_c_im = self.param(
            #     "power_c_im",
            #     lambda k, s: self.power_init_range*jax.random.normal(k, s),
            #     (self.power_law_modes,),
            # )

            if self.include_short_range:
                self.green_function_gate = self.param(
                    "green_function_gate",
                    lambda k, s: 0.0,
                    (1,),
                ) 
            


    def __call__(self, u, recurrent, cache_ssm):
        """ Applies the S4 convolution.
        Args:
            u: Input sequence of shape (l,) or (l, d_model/2) if complex_input=True.
               The dtype should match complex_input flag.
            mode: Processing mode (e.g., "full").
        Returns:
            y: Output sequence of shape (l,) or (l, d_model/2) complex.
        """

        if recurrent:
            return self.recurrent_mode(u)

        l = u.shape[0]
        assert l == self.l_max, f"Input length {l} does not match l_max {self.l_max} for S4Kernel instance"

        G_roots_complex = green_roots(
            self.Lambda, self.P, self.P, self.B, self.C, self.delta, self.l_max #Warning changed Q -> P
        )

        if self.normalize_s4: 
            G_roots_complex = self.scale_greens_function * G_roots_complex / jnp.linalg.norm(G_roots_complex) # Normalize the roots

        if self.use_power_law:
            green_function_power_law=  greens_function_power_law(
                self.power_alpha_raw, self.power_c_re, self.l_max
            )
            if not self.include_short_range:
                G_roots_complex = green_function_power_law
            else: 
                gate = nn.sigmoid(self.green_function_gate) # Extract scalar gate value
                G_roots_complex = gate* G_roots_complex + (1 - gate) * green_function_power_law

        current_G_roots = jnp.asarray(G_roots_complex[:l], dtype=self.complex_dtype) # Use roots corresponding to actual input length

        D = self.D[0]
        if self.bidirectional:
            D = 2*D 
            current_G_roots = current_G_roots + jnp.roll(current_G_roots[::-1], 1)

        if cache_ssm and self.is_mutable_collection('cache') and not recurrent:
            self._cache_ssm(current_G_roots)


        if self.is_mutable_collection('cache'):
            self.g_roots_debug.value = current_G_roots # Store for debugging

        if self.complex_input:
            assert u.ndim == 2, f"Expected input shape (l,) for S4Kernel instance, got {u.shape}"
            u = (u[:, 0] + 1j * u[:, 1]).astype(self.complex_dtype) # Convert to complex

            # if self.is_mutable_collection('cache'):
            #     self.s4_internal_u_debug.value = u

            y = circular_convolution_complex(u, current_G_roots).astype(self.complex_dtype)

            # if self.is_mutable_collection('cache'):
            #     self.s4_internal_y_debug.value = y

            y = y + D * u # D is (1,)

            # if self.is_mutable_collection('cache'):
            #     self.s4_internal_y_debug_after.value = y

            y_out = jnp.stack([y.real, y.imag], axis=-1).astype(self.real_dtype) # Output is complex
            return y_out 
        else:
            # u is real (l,), G_roots is complex (l,)
            assert u.ndim == 1, f"Expected input shape (l,) for S4Kernel instance, got {u.shape}"
            assert not jnp.iscomplexobj(u), "Input must be real when complex_input=False"

            # Use rfft for efficiency if output should be real.
            y_complex = circular_convolution_complex(u.astype(self.complex_dtype), current_G_roots).astype(self.complex_dtype)
            y = y_complex.real.astype(self.real_dtype) # Take real part as output must be real

            # Add D term
            y = y + D * u # D is (1,) float, u is float
            return y
        

    def _cache_ssm(self, g_roots):
        
        if self.use_circulant_slice: 
            self.circulant_slice.value =  circulant_matrix_slice(
                g_roots, self.l_width, self.l_max) 
        else: 
            self.ssm.value = self._build_ssm()
        #self.sow("cache", "ssm", self._build_ssm())      # cheap assignment

    def _build_ssm(self):

        A, B, C = discrete_DPLR(
            self.Lambda, self.P, self.P, self.B, self.C, self.delta, self.l_max
        )
        L0      = 2 * self.l_width + 1
        AL      = jnp.linalg.matrix_power(A, self.l_max)
        AdeltaL = jnp.linalg.matrix_power(A, self.l_max - L0 ) 
        return (A, AL, AdeltaL, B, C)
        
    def recurrent_mode(self, u): 
        """
        Applies the S4 convolution in recurrent mode.
        Args:
            u: Input sequence of shape (l,) or (l, d_model/2) if complex_input=True.
               The dtype should match complex_input flag.
        Returns:
            y: Output sequence of shape (l,) or (l, d_model/2) complex.
        """

        # transform the complex input 
        
        if self.complex_input:
            assert u.ndim == 2, f"Expected input shape (l,) for S4Kernel instance, got {u.shape}"
            u = (u[:, 0] + 1j * u[:, 1]).astype(self.complex_dtype) # Convert to complex
        else:
            assert u.ndim == 1, f"Expected input shape (l,) for S4Kernel instance, got {u.shape}"
            assert not jnp.iscomplexobj(u), "Input must be real when complex_input=False"

        assert u.shape[0] == 2 * self.l_width + 1, \
            f"Input length {u.shape[0]} does not match l_width {self.l_width} for S4Kernel instance"

        if self.is_mutable_collection('cache'):
                self.s4_internal_u_debug.value = u

        D = self.D[0] # D is a scalar (1,)

        if self.use_circulant_slice:
            circulant_slice = self.get_variable("cache", "circulant_slice")

            y = circulant_slice@u.astype(self.complex_dtype) # Use circulant matrix slice
    
            if self.bidirectional:
                D = 2 * self.D[0] # Double D for bidirectional
        else: 
            ssm = self.get_variable("cache", "ssm")
            y = scan_SSM_periodic(*ssm, u[jnp.newaxis, :].astype(self.complex_dtype)).reshape(-1) # Scan over the input sequence

        if self.is_mutable_collection('cache'):
            self.s4_internal_y_debug.value = y
            self.s4_internal_y_debug_after.value = y

        if self.complex_input:
            y = y +  D * u # D is (1,)
            y_out = jnp.stack([y.real, y.imag], axis=-1).astype(self.real_dtype) # Output is complex
        else: 
            y_out = y.real.astype(self.real_dtype) + D * u # D is (1,)

        return y_out 
    




    def get_poles(self):
        """
        Get the poles of the S4 kernel in both z-domain and s-domain.
        
        Returns:
            A tuple (z_poles, s_poles) where:
            - z_poles: Complex poles in z-domain (discrete time)
            - s_poles: Complex poles in s-domain (continuous time)
        """
        # Construct actual Lambda values (complex)
        Lambda = self.Lambda  # Already computed in setup as -abs(Lambda_re) + 1j * Lambda_im
        
        # Get the z-domain poles using the provided formula
        z_poles = get_pole_1d(self.Lambda, self.P, self.Q, self.delta)
        
        # Map to s-domain
        s_poles = map_to_splane(z_poles, self.delta)
        
        return z_poles, s_poles

S4Kernel_vmap = cloneLayer(S4Kernel)
