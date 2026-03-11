# Import jax 
import jax
import flax.linen as nn
import jax.numpy as jnp
import jax.typing as jt
from flax.core.frozen_dict import freeze, unfreeze
from typing import Callable, Optional, Tuple, Dict, Any
from functools import partial
from dysonnet.partial_evaluation import fix_tuple

#Other
from einops import repeat 
from dysonnet.partial_evaluation import ActivationMean, ActivationMeanOutside, ActivationDifference, unwrap, sum_partial, difference_partial, slice_array_jit

from dysonnet.link_tensors import get_green_functions, get_D_from_params,get_D_from_params_fast, compute_link_tensors_real, link_multiply_real
from dysonnet.DysonBlock import DysonBlock
#do nothing

@jax.jit
def stable_logcosh(x):
    # one‐liner, fully jit‐compatible, never produces inf
    two = jnp.asarray(2.0, dtype=x.dtype)
    return jnp.logaddexp(x, -x) - jnp.log(two)


class ActNorm(nn.Module):
    hidden_dim: int
    epsilon: float = 1e-6
    use_bias : bool = True 

    @nn.compact
    def __call__(self, x):
        # during model.init(…) this is True; thereafter False
        if self.is_initializing():
            # compute batch statistics
            reduction_axis = tuple(range(x.ndim - 1))
            mean = jnp.mean(x, axis=reduction_axis, keepdims=True)
            var  = jnp.var(x, axis=reduction_axis, keepdims=True)

            # build initializers *closing over* mean/var
            def scale_init(key, shape):
                return (1.0 / jnp.sqrt(var + self.epsilon)).reshape(shape)
            def bias_init(key, shape):
                return (-mean / jnp.sqrt(var + self.epsilon)).reshape(shape)
        else:
            # after init, use trivial inits
            scale_init = nn.initializers.ones

            if self.use_bias:
                bias_init  = nn.initializers.zeros

        scale = self.param('scale', scale_init, (self.hidden_dim,))
        if not self.use_bias: 
            return scale * x 
        
        bias  = self.param('bias',  bias_init,  (self.hidden_dim,))
        return scale * x + bias



class ResidualBlockDyson(nn.Module):
    mixer: Callable
    L : int
    partial_evaluation: bool = False  
    norm = nn.LayerNorm
    epsilon: float = 1e-6  # Default epsilon value
    block_index : int = 0
    width : int = 0
    hidden_dim : int = 0
    link_tensor_approximate : bool = False 
    use_bias : bool = True  
    dtype: Any = jnp.float32


    def setup(self):

        self.norm_x = ActNorm(name = "LayerNorm_x", epsilon = self.epsilon, hidden_dim = self.hidden_dim, use_bias = self.use_bias)

        self.norm_g = self.norm(name = "LayerNorm_g", epsilon=self.epsilon, use_bias = self.use_bias)
        
        if self.block_index == 0:
            self.layer_norm_diff = ActivationDifference(name="layer_norm_diff", width=self.width, cache_dtype=self.dtype)
            self.input_diff  = ActivationDifference(name="input_diff", width=self.width, cache_dtype=self.dtype)
            
    def __call__(self, x, g, centers = None, mode: str = "full", link_buffer_mode : str = "", link_buffer = None, interlayer_link_buffer=None):
        self.sow("debug", "x_partial input", x)
       
        # Do the layer norm 
        h = self.norm_g(g)
        y = self.norm_x(x)
        


        if mode == "partial" and self.block_index > 0 and self.use_bias: 
            y = y - self.get_variable("params", "LayerNorm_x")["bias"]  # Subtract the bias for partial evaluation
        
        if self.block_index > 0 and self.link_tensor_approximate and mode == "partial":
            update = interlayer_link_buffer.at[self.block_index-1].get() 
            update_buffer = self.norm_x(update)
            if self.use_bias:
                update_buffer = update_buffer - self.get_variable("params", "LayerNorm_x")["bias"]
            interlayer_link_buffer = interlayer_link_buffer.at[self.block_index-1].set(update_buffer)
            self.sow("debug", "interlayer link norm", interlayer_link_buffer)
        if self.block_index == 0:
            y = self.layer_norm_diff(y, centers=centers, mode=mode)  # Apply the activation difference
            x = self.input_diff(x, centers=centers, mode=mode)  # Apply the activation difference

        if self.partial_evaluation and mode == "full": 
            self.sow("intermediates", "residual", x) # Store the residual for debugging
            self.sow("intermediates", "residual_g", g) # Store the residual for debugging
            
            jac_ln = self.norm_x_jacobian(x)
            self.sow("intermediates", "ln_jac_x", jac_ln)

        if link_buffer_mode == "final":
            assert self.partial_evaluation
            y, h, link_buffer = self.mixer(y, h, centers=centers, mode=mode, link_buffer_mode=link_buffer_mode, link_buffer=link_buffer)
            
            return x + h, h + g, link_buffer, interlayer_link_buffer
        elif link_buffer_mode == "interlayer":
            assert self.partial_evaluation
            self.sow("debug", "partial_before_mixer", y)
            y, h, link_buffer, interlayer_link_buffer = self.mixer(y, h, centers=centers, mode=mode, link_buffer_mode=link_buffer_mode, link_buffer=link_buffer, interlayer_link_buffer=interlayer_link_buffer)
            self.sow("debug", "partial_after_mixer", y)


            return x + y, h + g, link_buffer, interlayer_link_buffer
        else:
            y, h = self.mixer(y, h, mode=mode)
            return x + y, h + g, link_buffer, interlayer_link_buffer

    def norm_x_jacobian(self, x):
        jac_fwd= jax.jacfwd(self.norm_x)
        # x has shape (B, L, D), so we need to vmap over batch and length
        jac_tokens = jax.vmap(jac_fwd)      # vmap over L (second axis)
        jac_batch = jax.vmap(jac_tokens)    # vmap over B (first axis)

        return jac_batch(x)
        
    def forward_and_pullback(self, x, g):
        """
        Forward pass and pullback for the GatedResidualBlock.
        
        Args:
            x: Input tensor
            g: Gating tensor
            
        Returns:
            Output tensor after processing through the block.
        """

        if self.partial_evaluation:
            jac_ln = self.norm_x_jacobian(x)
            self.sow("intermediates", "ln_jac_x", jac_ln)

        # Do the layer norm 
        h = self.norm_g(g)
        
        y, pullback_norm = jax.vjp(self.norm_x, x)
        
        self.sow("debug", "x", x)
        if self.block_index == 0:
            self.layer_norm_diff(y, mode="full")
            self.input_diff(x, mode="full")
        y, h, pullbacks = self.mixer.forward_and_pullback(y, h, mode="full")
        

        # Update the pullbacks with the layer norm 
        p_bs4, p_s4, p_a_s4 = pullbacks 
        
        def p_bs4_new(x, p_bs4_fn = p_bs4):
            return unwrap(pullback_norm(p_bs4_fn(x)))
        
        pullbacks_new = (p_bs4_new, p_s4, p_a_s4)

        return x + y, h + g, pullbacks_new
    

    def forward_partial(self, x_full, x_partial, g_full, g_partial, link_buffer, skip_buffer, centers):
        """
        Forward pass and pullback for the GatedResidualBlock.
        
        Args:
            x: Input tensor
            g: Gating tensor
            
        Returns:
            Output tensor after processing through the block.
        """
        # Do the layer norm 
        h_full = self.norm_g(g_full)
        y_full, pullback_norm = jax.vjp(self.norm_x, x_full)

        self.sow("debug", "x_partial input", x_partial)
        # Partial mode 
        h_partial = self.norm_g(g_partial) 
        y_partial = self.norm_x(x_partial) 



        # ——— mirror what __call__ does for partial at block>0 ———
        if self.block_index > 0 and self.use_bias:

            y_partial = y_partial - self.get_variable("params", "LayerNorm_x")["bias"]

        
        if self.block_index == 0: 
            y_partial = difference_partial(y_full, y_partial, centers, self.width)
            x_partial = difference_partial(x_full, x_partial, centers, self.width) 

        if self.block_index > 0 and self.link_tensor_approximate:
            skip_buffer = self.norm_x(skip_buffer)
            if self.use_bias:
                skip_buffer = skip_buffer - self.get_variable("params", "LayerNorm_x")["bias"]
            self.sow("debug", "interlayer link norm", skip_buffer)


        # ----- Call the token mixer with the partial inputs -----
        self.sow("debug", "partial_before_mixer", y_partial)
        y_full, y_partial, h_full, h_partial, link_buffer, skip_buffer, pullbacks = self.mixer.forward_partial(y_full, y_partial, h_full, h_partial, link_buffer, skip_buffer, centers)
        self.sow("debug", "partial_after_mixer", y_partial)

        # Update the pullbacks with the layer norm 
        p_bs4, p_s4, p_a_s4 = pullbacks 
        
        def p_bs4_new(x, p_bs4_fn = p_bs4):
            return unwrap(pullback_norm(p_bs4_fn(x)))
        
        pullbacks_new = (p_bs4_new, p_s4, p_a_s4)

        return x_full+y_full, x_partial+y_partial, h_full+g_full, h_partial+g_partial, link_buffer, skip_buffer, pullbacks_new


    def compute_moments(self, x, input_layer, activation_layer, use_residual, norm_offset):
        # Do the layer norm
        y = self.norm_x(x) 

        # Conditionally remove LayerNorm bias when norm_offset is False
        if self.use_bias:
            offset = self.get_variable("params", "LayerNorm_x")["bias"]
            y = jnp.where(norm_offset[:, None, None], y, y - offset)

        y = self.mixer.compute_moments(y, input_layer, activation_layer)
        
        # dynamically add residual based on use_residual array
        return jnp.where(use_residual[:, None, None], x + y, y)
 


    def sow_jacobian(self, jacobian, include_s4 : bool = True): 
        self.mixer.sow_jacobian(jacobian, include_s4=include_s4)

    def sow_link_tensor(self, link_tensor):
        return self.mixer.sow_link_tensor(link_tensor)

class LogCoshLayer(nn.Module):
    """
    Implements the sum_alpha log(cosh(b_alpha + w_alpha·z)) activation function.
    
    This layer implements the function:
        f(z; W) = sum_alpha log(cosh(b_alpha + w_alpha·z))
    
    where b_alpha and w_alpha are learnable parameters.
    
    Attributes:
        num_hidden: Number of hidden neurons (K in the paper)
        use_complex: Whether to use complex-valued parameters
        dtype: Data type for parameters (defaults to float32 or complex64)
    """
    num_hidden: int
    L : int 
    
    use_complex: bool = False
    width : int = 0
    dtype: Optional[Any] = None
    embedding_dim : Optional[int] = None
    marshall_rule : bool = False

    def setup(self):
        # Determine data type based on complex flag
        if self.dtype is None:
            self.param_dtype = jnp.complex64 if self.use_complex else jnp.float32
        else:
            self.param_dtype = self.dtype
            
        # Initialize biases here in setup
        self.weights = self.param(
                                'weights',
                                lambda key, shape: self._init_params(key, shape),
                                (self.num_hidden, self.embedding_dim),
        )

        self.biases = self.param(
            'biases', 
            lambda key: self._init_params(key, (self.num_hidden,))
        )


        
    def _init_params(self, key, shape):
        """Initialize parameters with appropriate dtype."""
        if self.use_complex:
            key1, key2 = jax.random.split(key)
            real_dtype = jnp.empty((), self.param_dtype).real.dtype
            real_part = jax.random.normal(key1, shape, dtype=real_dtype) * 0.01
            imag_part = jax.random.normal(key2, shape, dtype=real_dtype) * 0.01
            return (real_part + 1j * imag_part).astype(self.param_dtype)
        else:
            return jax.random.normal(key, shape, dtype=self.param_dtype) * 0.01

    @nn.compact
    def __call__(self, z, mode: str = "full", residual = 0, centers = 0, L :int =0, L0 : int = 0, mean_outside = 0, headless = False, input_config=None, skip_offset = True):
        """
        Apply the log-cosh activation function.
        
        Args:
            z: Input tensor of shape (..., features)
            mode: Processing mode (e.g., "full").
            
        Returns:
            Output tensor of shape (...) - scalar output for each batch item
        """

        weights = self.weights
        
        # Calculate linear transformation: b_alpha + w_alpha·z
        linear_output = self.biases + jnp.einsum('...i,ji->...j', z, weights)
        
        # Now we need to insert the outside field 
        if mode == "partial":
            L = self.L 
            L0 = 2* self.width + 1
            residual = jnp.einsum('...i,ji->...j', residual, weights)
            mean_outside = mean_outside + residual + self.biases
            linear_output = L0/L * linear_output +  mean_outside * (L - L0)/L 
        elif mode == "link": 
            self.sow("debug", "linear_output_before", linear_output)
            self.sow("debug", "mean_outside", mean_outside)
            linear_output = linear_output + mean_outside

        if headless:
            return linear_output 

        self.sow("debug", "linear_output", linear_output)

        # Apply log(cosh) activation function
        if self.use_complex or jnp.issubdtype(z.dtype, jnp.complexfloating):
            neg_2x = -2 * linear_output
            result = linear_output + jnp.log(1 + jnp.exp(neg_2x)) - jnp.log(2.0)
        else:
            result = stable_logcosh(linear_output)

        # Sum over the hidden dimension (alpha)
        output = jnp.sum(result, axis=-1, dtype=self.param_dtype)
        
        if self.marshall_rule:
            n_sublattice = (1 - input_config[:, ::2]) / 2  # Convert to 0/1 occupancy
            phase = (1j * jnp.pi * jnp.sum(n_sublattice, axis=-1)).astype(self.param_dtype) # Sum over length
            output = output + phase 

        return output   

    def forward_headless(self, z):
        weights = self.weights
        
        # Calculate linear transformation: b_alpha + w_alpha·z
        return self.biases + jnp.einsum('...i,ji->...j', z, weights)

    def forward_partial(self, z_full, z_partial, link_buffer, headless = False):
        """
        Apply the log-cosh activation function.
        
        Args:
            z: Input tensor of shape (..., features)
            mode: Processing mode (e.g., "full").
            
        Returns:
            Output tensor of shape (...) - scalar output for each batch item
        """

        weights = self.weights
        biases = self.biases
        
        # Calculate linear transformation: b_alpha + w_alpha·z
        linear_output_full    = jnp.einsum('...i,ji->...j', z_full, weights) + biases
        linear_output_partial = jnp.einsum('...i,ji->...j', z_partial, weights) + biases

        self.sow("debug", "linear_output_before", linear_output_partial)
        self.sow("debug", "mean_outside", link_buffer)

        linear_output_partial = linear_output_partial + link_buffer  

        if headless:
            return linear_output_full, linear_output_partial

        self.sow("debug", "linear_output", linear_output_partial)

        # Apply log(cosh) activation function
        if self.use_complex or jnp.issubdtype(z_full.dtype, jnp.complexfloating):
            neg_2x_full = -2 * linear_output_full
            result_full = linear_output_full + jnp.log(1 + jnp.exp(neg_2x_full)) - jnp.log(2.0)

            neg_2x_partial = -2 * linear_output_partial
            result_partial = linear_output_partial + jnp.log(1 + jnp.exp(neg_2x_partial)) - jnp.log(2.0)
        else:
            result_full = stable_logcosh(linear_output_full)
            result_partial = stable_logcosh(linear_output_partial) 

        # Sum over the hidden dimension (alpha)
        output_full = jnp.sum(result_full, axis=-1, dtype=self.param_dtype)
        output_partial = jnp.sum(result_partial, axis=-1, dtype=self.param_dtype) 
        
        return output_full, output_partial   


    def get_pullback(self, z):
        _, pullback = jax.vjp(lambda zz : self.forward_headless(jnp.mean(zz, axis=1)), z)
        return pullback
    
    def compute_moments(self, x):
        return jnp.einsum('...i,ji->...j', x, self.weights)  
    

class DysonNet(nn.Module):
    """
    DysonNet Neural Quantum States model combining gated S4 blocks with fixed LogCosh output.
    
    This class implements a complete architecture using S4-based blocks for quantum state representation.
    It stacks multiple blocks with residual connections and provides flexible configuration options.
    
    Attributes:
        token_size: Size of input tokens
        embedding_dim: Dimension for token embeddings
        n_blocks: Number of blocks to stack
        hidden_dim: Dimension for hidden representations (default: embedding_dim)
        s4_config: Configuration dictionary for S4 blocks (optional)
        dropout_rate: Dropout rate for regularization (default: 0.0)
        use_gated_mlp: Whether to use gated MLP feed-forward blocks (default: True)
        use_logcosh_output: Fixed to True (LogCosh output is always used)
        logcosh_hidden: Number of hidden neurons for LogCosh layer (default: 64)
        use_complex: Whether to use complex-valued parameters in the LogCosh layer (default: False)
        bidirectional: Whether to use bidirectional S4 processing (default: False)
        s4_init_type: Initialization method for S4 blocks ("hippo", "random")
        s4_decay_scale: Scale for eigenvalue real parts in random S4 initialization
        s4_freq_scale: Scale for eigenvalue imaginary parts in random S4 initialization
        s4_long_range_frac: Fraction of long-range modes in random S4 initialization
    """
    token_size: int
    embedding_dim: int
    n_blocks: int
    hidden_dim: Optional[int] = None
    s4_config: Optional[Dict[str, Any]] = None
    seq_length: int = 128  # Default sequence length
    dropout_rate: float = 0.0
    use_gated_mlp: bool = True
    use_logcosh_output: bool = True
    logcosh_hidden: int = 64
    use_complex: bool = False
    complex_output : bool = False
    bidirectional: bool = False
    use_convolution: bool = True
    conv_layer_number : int = 1
    use_embedding_bias : bool = True
    embedding_type: str = "linear"  # "linear" (Dense on raw spins) or "full_rank" (one-hot over 2^token_size)
    shift_project : bool = False

    # S4 initialization parameters
    s4_init_type: str = "hippo"  # Options: "hippo", "random"
    s4_decay_scale: float = 1.0  # For random initialization
    s4_freq_scale: float = 1.0  # For random initialization
    s4_long_range_frac: float = 0.5  # For random initialization
    s4_include_interblock : bool = False 
    s4_use_circulant_slice: bool = True  # Whether to use circulant slice in S4 blocks
    s4_normalize : bool = False 
    use_layer_norm_mixer : bool = False 
    use_symmetric_conv: bool = False  # Whether to use symmetric convolution in S4 blocks
    link_tensor_approximate : bool = False
    marshall_rule : bool = False

    real_dtype: Any = jnp.float32
    complex_dtype: Any = jnp.complex64

    # S4 power law 
    use_power_law : bool = False 
    power_law_modes : int = 1 
    power_init_range : float = 2.0# Range for power law initialization
    include_short_range : bool = False

    partial_evaluation : bool = False 
    s4_l_width: Optional[int] = None

    def setup(self):
        """Initialize the DysonNet model components."""
        # Use embedding_dim as hidden_dim if not specified
        hidden_dim = self.hidden_dim or self.embedding_dim 
        
        # Default configuration for S4 blocks
        default_s4_config = {
            "l_width": self.s4_l_width,
            "d_state": 8,
            "l_max": self.seq_length,
            "use_conv": self.use_convolution,
            "conv_kernel_size": 4,
            "conv_layer_number" : self.conv_layer_number,
            "conv_use_bias": True,
            "use_gating": False,
            "bidirectional": self.bidirectional,
            "complex_params": self.use_complex,
            "dropout_rate": self.dropout_rate,
            "init_type": self.s4_init_type,
            "decay_scale": self.s4_decay_scale,
            "freq_scale": self.s4_freq_scale,
            "long_range_frac": self.s4_long_range_frac,
            "use_circulant_slice": self.s4_use_circulant_slice,  # Default to False, can be overridden in s4_config
            "use_layer_norm_mixer" : self.use_layer_norm_mixer,
            "use_symmetric_conv": self.use_symmetric_conv,  # Default to False, can be overridden in s4_config
            "link_tensor_approximate" : self.link_tensor_approximate, 
            "use_power_law" : self.use_power_law,
            "power_law_modes" : self.power_law_modes,
            "power_init_range" : self.power_init_range,
            "include_short_range" : self.include_short_range, 
            "s4_normalize": self.s4_normalize,  # Default to True, can be overridden in s4_config
        }

        if self.embedding_type not in ("linear", "full_rank"):
            raise ValueError(f"embedding_type must be 'linear' or 'full_rank', got {self.embedding_type}")

        if not self.use_logcosh_output:
            raise ValueError("use_logcosh_output is fixed to True for DysonNet.")

        embedding_use_bias = self.use_embedding_bias
        
        # Merge default configs with provided configs
        s4_config = freeze({**default_s4_config, **(self.s4_config or {})})
        # Embedding layers. Keep the linear path as the default/backward-compatible option.
        if self.embedding_type == "linear":
            self.embedding = nn.Dense(self.embedding_dim, dtype=self.real_dtype, use_bias=embedding_use_bias)
        else:
            self.embedding_full_rank = nn.Dense(self.embedding_dim, dtype=self.real_dtype, use_bias=embedding_use_bias)

        # Precompute constants for full-rank embedding; harmless for linear and avoids tracing-time work.
        self._token_powers = jnp.asarray(
            2 ** jnp.arange(self.token_size - 1, -1, -1, dtype=jnp.int32),
            dtype=jnp.int32,
        )
        
        # Create gated S4 blocks
        blocks = []
        for i in range(self.n_blocks):
            block = DysonBlock(
                d_model=hidden_dim,
                l_max=s4_config["l_max"],
                l_width=s4_config["l_width"],
                d_state=s4_config["d_state"],
                use_conv=s4_config["use_conv"],
                conv_layer_number=s4_config["conv_layer_number"],
                conv_kernel_size=s4_config["conv_kernel_size"],
                conv_use_bias=s4_config["conv_use_bias"],
                use_gating=s4_config["use_gating"],
                bidirectional=s4_config["bidirectional"],
                complex_params=s4_config["complex_params"],
                dtype=self.real_dtype,
                complex_dtype=self.complex_dtype,
                dropout_rate=s4_config["dropout_rate"],
                init_type=s4_config["init_type"],
                decay_scale=s4_config["decay_scale"],
                freq_scale=s4_config["freq_scale"],
                long_range_frac=s4_config["long_range_frac"],
                partial_evaluation=self.partial_evaluation,
                include_interblock=self.s4_include_interblock,
                use_circulant_slice=self.s4_config.get("use_circulant_slice", True),
                block_index=i,
                use_layer_norm_mixer=s4_config["use_layer_norm_mixer"],
                use_symmetric_conv=s4_config.get("use_symmetric_conv", False),  # Use symmetric convolution by default
                link_tensor_approximate=self.link_tensor_approximate,  # Use link tensor correction by default
                use_power_law=s4_config.get("use_power_law", False),
                power_law_modes=s4_config.get("power_law_modes", 1),
                power_init_range=s4_config.get("power_init_range", 2.0),
                include_short_range=s4_config.get("include_short_range", False),
                normalize_s4=s4_config.get("s4_normalize", False),  # Default to True, can be overridden in s4_config
            )

            blocks.append(ResidualBlockDyson(
                mixer=block,
                L=self.seq_length,
                partial_evaluation=self.partial_evaluation,
                block_index=i,
                width=s4_config["l_width"],
                hidden_dim=hidden_dim,
                link_tensor_approximate=self.link_tensor_approximate,
                dtype=self.real_dtype,
            ))


        self.blocks = blocks
        
        # Fixed LogCosh output (no final MLP).
        self.logcosh = LogCoshLayer(
            num_hidden=self.logcosh_hidden,
            embedding_dim=self.embedding_dim,
            use_complex=self.complex_output,
            dtype=self.complex_dtype if self.complex_output else self.real_dtype,
            L=self.seq_length,
            width=self.s4_l_width,
        )

        if self.partial_evaluation: 
            #self.mean_initial = ActivationMeanOutside(name = "mean_initial", L = self.seq_length, width = self.s4_l_width)
            self.mean_final = ActivationMeanOutside(name = "mean_final", L = self.seq_length, width = self.s4_l_width)
            self.input_diff = ActivationDifference(name = "input_diff", width = self.s4_l_width)

    def _tokens_to_indices(self, tokens: jax.Array) -> jax.Array:
        """
        Map ±1 spin tokens of shape (..., token_size) to integer indices in [0, 2^token_size - 1].
        Bits are ordered from left→right: idx = sum bit[k] * 2^(token_size-1-k).
        """
        tokens = jnp.asarray(tokens, dtype=self.real_dtype)
        bits = ((1.0 - tokens) * 0.5).astype(jnp.int32)  # +1 -> 0, -1 -> 1
        return jnp.sum(bits * self._token_powers, axis=-1)

    def _embedding_baseline_tokens(self, x_tokens: jax.Array) -> jax.Array:
        """Baseline token used to compute an embedding offset."""
        if self.embedding_type == "full_rank":
            # All +1 spins map to the zero one-hot index.
            return jnp.ones_like(x_tokens)
        return jnp.zeros_like(x_tokens)

    def _run_full_rank_sanity_checks(self, one_hot: jax.Array, embedded: jax.Array) -> None:
        """Static shape sanity checks; cheap and JIT-friendly."""
        if self.embedding_type != "full_rank":
            return

        expected_dim = self._num_token_configs
        assert one_hot.shape[-1] == expected_dim, "One-hot dimension mismatch for full-rank embedding"
        assert embedded.shape[-1] == self.embedding_dim, "Embedding output has wrong feature dimension"

    def _embed_tokens(self, x_tokens: jax.Array) -> jax.Array:
        """Dispatch embedding based on embedding_type."""
        if self.embedding_type == "linear" or self.token_size == 1:
            return self.embedding(x_tokens)

        indices = self._tokens_to_indices(x_tokens)
        one_hot = jax.nn.one_hot(indices, self._num_token_configs, dtype=self.real_dtype)
        embedded = self.embedding_full_rank(one_hot)
        self._run_full_rank_sanity_checks(one_hot, embedded)
        return embedded

    def __call__(self, x, centers = None, train: bool = True, mode: str = "full", cache_jac = False, headless = False, z2_project=False, remove_shift=False):
        """
        Forward pass of the DysonNet model.
        
        Args:
            x: Input tensor
            train: Whether in training mode (affects dropout)
            mode: Processing mode (e.g., "full").
            
        Returns:
            Output tensor after processing through the model.
            Shape depends on output mode:
            - LogCosh: (batch_size,)
            - LatentMPS: (batch_size, seq_length, 2, chi, chi)

        Mode combinations (high-level):
        - Full path: `partial_evaluation=False` OR `cache_jac=True`
        - Partial path: `partial_evaluation=True` AND `cache_jac=False`
        - `mode="partial"` should only be used when `partial_evaluation=True`
        """

        if centers is not None:
            idx, c = centers 
            c = c // self.token_size 
            centers = (idx, c)

        if z2_project:
            if mode == "partial":
                raise ValueError("Z2 projection not supported in partial mode.")
            psi = self._dyson_forward(x, centers=centers, train=train, mode=mode, cache_jac=cache_jac, headless=headless)
            psi_flipped = self._dyson_forward(-x, centers=centers, train=train, mode=mode, cache_jac=cache_jac, headless=headless)
            return jnp.logaddexp(psi, psi_flipped) - jnp.log(2.0)
        
        if self.shift_project and not remove_shift: 

            psi = self._dyson_forward(x, centers=centers, train=train, mode=mode, cache_jac=cache_jac, headless=headless)
            
            x_shift = jnp.roll(x, 1, axis=-1)
            psi_flipped = self._dyson_forward(x_shift, centers=centers, train=train, mode=mode, cache_jac=cache_jac, headless=headless)
            return jnp.logaddexp(psi, psi_flipped) - jnp.log(2.0)


        return self._dyson_forward(x, centers=centers, train=train, mode=mode, cache_jac=cache_jac, headless=headless)


    def _dyson_forward(self, x, centers = None, train: bool = True, mode: str = "full", cache_jac = False, headless = False):
        """
        Forward pass of the DysonNet model.
        
        Args:
            x: Input tensor
            train: Whether in training mode (affects dropout)
            mode: Processing mode (e.g., "full").
            
        Returns:
            Output tensor after processing through the model.
            Shape depends on output mode:
            - LogCosh: (batch_size,) 
            - LatentMPS: (batch_size, seq_length, 2, chi, chi)
        """
        if self.partial_evaluation and not cache_jac:
            return self._dyson_forward_partial(
                x,
                centers=centers,
                train=train,
                mode=mode,
                headless=headless,
            )

        return self._dyson_forward_full(
            x,
            centers=centers,
            train=train,
            mode=mode,
            cache_jac=cache_jac,
            headless=headless,
        )

    def _dyson_forward_full(self, x, centers = None, train: bool = True, mode: str = "full", cache_jac = False, headless = False):
        """
        Forward pass for full-mode evaluation (including optional link tensor caching).

        Valid combinations:
        - `mode="full"` always supported.
        - `mode="partial"` should only be used when `partial_evaluation=True`.
        - `cache_jac=True` triggers Jacobian/pullback caching (link tensors) in full mode.
        """
        # Reshape input into token sequence
        x_input = x
        batch_size = x.shape[0]
        x = x.astype(self.real_dtype)  # Ensure input matches configured dtype
        x = jnp.reshape(x, (batch_size, -1, self.token_size))

        # Apply embedding
        x = self._embed_tokens(x)

        g = x
        if cache_jac:
            x, g, pullback = self.forward_with_pullbacks_blocks(x)  # Get pullbacks for each block
        else:
            for block in self.blocks:
                x, g, _, _ = block(x, g, mode=mode, centers=centers)  # Pass mode here

        if self.partial_evaluation and cache_jac and mode == "full":
            # Compute the pullbacks
            final_pullback = self.logcosh.get_pullback(x)  # Get pullback for the final LogCosh layer
            self._get_jacobians_pullbacks(x, pullback, final_pullback)  # Store pullbacks for each block

            # Compute two layer link tensors
            if not self.link_tensor_approximate:
                self._cache_link_tensors()

        return self._dyson_finalize_output(
            x,
            x_input,
            centers=centers,
            mode=mode,
            headless=headless,
            link_buffer=None,
        )

    def _dyson_forward_partial(self, x, centers = None, train: bool = True, mode: str = "full", headless = False):
        """
        Forward pass for partial evaluation / local updates using cached link tensors.

        Preconditions:
        - `partial_evaluation=True`
        - `cache_jac=False` (handled by caller)
        - `mode="partial"` expected for local updates
        """
        # Reshape input into token sequence
        x_input = x
        batch_size = x.shape[0]
        x = x.astype(self.real_dtype)  # Ensure input matches configured dtype
        x = jnp.reshape(x, (batch_size, -1, self.token_size))

        # Apply embedding
        x = self._embed_tokens(x)

        g = x
        link_buffer = jnp.zeros((x.shape[0], self.logcosh_hidden), dtype=self.logcosh.weights.dtype)  # Initialize mean outside

        interlayer_link_buffer = None
        link_buffer_mode = "final"
        if self.s4_include_interblock:
            # Allocate buffer for interlayer skip connections
            interlayer_link_buffer = jnp.zeros((len(self.blocks) - 1,) + x.shape, dtype=x.dtype)
            link_buffer_mode = "interlayer"
        else:
            self.sow("debug", "mini manage x ", x)

        for j, block in enumerate(self.blocks, start=1):
            # Deal with the gated residual block
            assert isinstance(block, (ResidualBlockDyson)), "Blocks must be of type GatedResidualBlock"
            x, g, link_buffer, interlayer_link_buffer = block(
                x,
                g,
                centers=centers,
                mode=mode,
                link_buffer_mode=link_buffer_mode,
                link_buffer=link_buffer,
                interlayer_link_buffer=interlayer_link_buffer,
            )
            if self.s4_include_interblock:
                self.sow("debug", f"x_block{j}", x)

        return self._dyson_finalize_output(
            x,
            x_input,
            centers=centers,
            mode=mode,
            headless=headless,
            link_buffer=link_buffer,
        )

    def _dyson_finalize_output(self, x, x_input, centers = None, mode: str = "full", headless = False, link_buffer = None):
        """
        Shared tail: pool, apply mean/outside correction, then LogCosh.

        Notes:
        - In `mode="partial"`, `link_buffer` must be provided.
        - In `mode="full"`, `link_buffer` is ignored.
        """
        if mode == "full":
            assert self.seq_length == x.shape[1], "Sequence length mismatch in full mode"

        x = jnp.sum(x, axis=1) / self.seq_length

        if self.partial_evaluation:
            if mode == "full":
                self.sow("intermediates", "mean_0", x)
            else:
                idx, _ = centers
                x0 = unwrap(self.get_variable("intermediates", "mean_0"))
                x = x + x0[idx, ...]  # Add the mean outside to the final output
                self.sow("debug", "final_mean", x)
                self.sow("debug", "final_link_buffer", link_buffer)
                #x = x + link_buffer

        if mode == "partial":
            self.sow("debug", "link_update", link_buffer)
            x = self.logcosh(x, mode="link", residual=0, mean_outside=link_buffer, headless=headless, input_config=x_input)
        elif mode == "full":
            x = self.logcosh(x, headless=headless, input_config=x_input)

        return x
    


    def _compute_moments(self, x, g_vals, gate_vals, use_residual_connections, norm_offset, pass_input):
        """
        Forward pass of the DysonNet model.
        
        Args:
            x: Input tensor
            train: Whether in training mode (affects dropout)
            mode: Processing mode (e.g., "full").
            
        Returns:
            Output tensor after processing through the model.
            Shape depends on output mode:
            - LogCosh: (batch_size,) 
            - LatentMPS: (batch_size, seq_length, 2, chi, chi)
        """
        
        x_input = x 
        batch_size = x.shape[0]
        x = x.astype(self.real_dtype)  # Ensure input matches configured dtype
        x = jnp.reshape(x, (batch_size, -1, self.token_size))

        # Apply embedding
        baseline_tokens = self._embedding_baseline_tokens(x)
        offset = self._embed_tokens(baseline_tokens)
        x = self._embed_tokens(x)
        
        x = jnp.where(pass_input[:, None, None], x - offset, x)
    
        for j, block in enumerate(self.blocks):
            assert isinstance(block, ResidualBlockDyson)
            x = block.compute_moments(x, g_vals[j], gate_vals[j], use_residual_connections[j], norm_offset[j])

        assert self.seq_length == x.shape[1], "Sequence length mismatch in full mode"
            
        x = jnp.sum(x, axis=1) / self.seq_length
        x = self.logcosh.compute_moments(x)
        return x


    def forward_with_pullbacks_blocks(self, x):
        """
        Returns:
          x_final : output of the last block
          g_final : final auxiliary value (your 'g')
          pullbacks: tuple with one pullback per block, in forward order
        """

        g = x 
        pullbacks = {}
        for j, block in enumerate(self.blocks):
            x, g, pullback = block.forward_and_pullback(x, g)
            pullbacks[j] = pullback
        
        return x, g, pullbacks  # Return the final output and all pullbacks


    
    def _get_jacobians_pullbacks(self, x, pullbacks, final_pullback):
        b= x.shape[0]  # Get the dimension of the input
        d_out = self.logcosh_hidden
        dtype = x.dtype
        dtype_seed = self.logcosh.weights.dtype
        inputs = jnp.broadcast_to(
            jnp.eye(d_out, d_out, dtype=dtype_seed), (b, d_out, d_out)
        )
        inputs = unwrap(
            jax.vmap(final_pullback, in_axes=-1, out_axes=-1)(inputs)
        )
        inputs = inputs.astype(dtype)
        for i in range(len(self.blocks), 0, -1):
            block = self.blocks[i-1]
            p_b_s4, p_s4, p_a_s4 = pullbacks[i-1]
            
            inputs_residual = inputs  # Store the residual for this block
            inputs = inputs.astype(dtype)
            inputs = jax.vmap(p_a_s4, in_axes=-1, out_axes=-1)(inputs) 
            block.sow_jacobian(jnp.einsum('bldk-> bkld', inputs), include_s4=False)  # Store the Jacobian for this block
            
            # Now push it back before s4 
            inputs = inputs.astype(dtype)
            inputs = jax.vmap(p_s4, in_axes=-1, out_axes=-1)(inputs)  
            block.sow_jacobian(jnp.einsum('bldk-> bkld', inputs), include_s4=True)  # Store the Jacobian for this block 
            
            # Push it to the beginning (and take care of the residual connections)
            inputs = inputs.astype(dtype)
            inputs = jax.vmap(p_b_s4, in_axes=-1, out_axes=-1)(inputs)
            inputs = inputs + inputs_residual
            
            self.sow("debug", f"inputs_end_loop{i}", inputs) 


    def _apply_jacobians_pullbacks(self, x, x_partial_buffer, centers, pullbacks, final_pullback):
        b= x.shape[0]  # Get the dimension of the input
        d_out = self.logcosh_hidden
        dtype = x.dtype
        dtype_seed = self.logcosh.weights.dtype
        inputs = jnp.broadcast_to(
            jnp.eye(d_out, d_out, dtype=dtype_seed), (b, d_out, d_out)
        )
        inputs = unwrap(
            jax.vmap(final_pullback, in_axes=-1, out_axes=-1)(inputs)
        )
        inputs = inputs.astype(dtype)

        x_partial_update = jnp.zeros((x_partial_buffer.shape[1], d_out), dtype=dtype)

        for i in range(len(self.blocks), 0, -1):
            block = self.blocks[i-1]
            p_b_s4, p_s4, p_a_s4 = pullbacks[i-1]
            
            inputs_residual = inputs  # Store the residual for this block
            inputs = inputs.astype(dtype)
            inputs = jax.vmap(p_a_s4, in_axes=-1, out_axes=-1)(inputs) 
            
            # Now push it back before s4 
            inputs = inputs.astype(dtype)
            inputs = jax.vmap(p_s4, in_axes=-1, out_axes=-1)(inputs)  
            
            jacobian = jnp.einsum('bldk-> bkld', inputs)  
            jac_with_s4 = slice_array_jit(jacobian, *centers, width = self.s4_l_width, axis = 1)
            
            self.sow("debug", f"jac_with_s4_sliced_{i}", jac_with_s4)  # Store the Jacobian with S4 for debugging
            self.sow("debug", f"jac_with_s4_input_{i}", x_partial_buffer[i-1])
            update = jnp.einsum('bkld, bld -> bk', jac_with_s4, x_partial_buffer[i-1])
            x_partial_update = x_partial_update + update 

            # Push it to the beginning (and take care of the residual connections)
            inputs = inputs.astype(dtype)
            inputs = jax.vmap(p_b_s4, in_axes=-1, out_axes=-1)(inputs)
            inputs = inputs + inputs_residual
            
            self.sow("debug", f"inputs_end_loop{i}", inputs) 
            #print("input end loop ", inputs.shape)

        return x_partial_update

    def _cache_link_tensors(self):
        params = self.variables['params']
        intermediates = self.variables['intermediates']
        cache = self.variables["cache"]
        
        w = self.s4_l_width
        delta_j_range = jnp.arange(-w, w + 1)  # Range for delta_j

        for j in range(len(self.blocks)-1):
            # Compute link tensors
            groots_1_rs, groots_2_rs = get_green_functions(params, cache, block_index=j)
            self.sow("debug", f"groots_2_rs_block_{j}", groots_2_rs)
            D, D_conj = get_D_from_params_fast(params, intermediates, block_index=j) 
            link_tensor = compute_link_tensors_real(D, D_conj, groots_1_rs, groots_2_rs, delta_j_range, symmetric=False)
            # Sow link tensor 
            self.blocks[j].sow_link_tensor(link_tensor)  # Store the link tensor in the block


    @jax.profiler.annotate_function
    @partial(jax.jit, static_argnames=["block_index"])
    def _apply_link_tensors(self, x_partial, centers, block_index : int):
        self.sow("debug", "link tensor input", x_partial) # Store the S4 input for debugging

        params = self.variables['params']
        intermediates = self.variables['intermediates']
        cache = self.variables["cache"]
        
        w = self.s4_l_width
        delta_j_range = jnp.arange(-w, w + 1)  # Range for delta_j

        j = block_index

        # Compute link tensors
        groots_1_rs, groots_2_rs = get_green_functions(params, cache, block_index=j, second_cached=False, l_max = self.seq_length, bidirectional = self.bidirectional)
        self.sow("debug", f"groots_2_rs_block_{j}", groots_2_rs)
        D, D_conj = get_D_from_params_fast(params, intermediates, block_index=j) 
        link_tensor = compute_link_tensors_real(D, D_conj, groots_1_rs, groots_2_rs, delta_j_range, symmetric=False)
        
        self.sow("debug", f"link_tensor_block_{j}", link_tensor)
        self.sow("debug", f"link_tensor_input_actual_{j}", x_partial)
        link_tensor = slice_array_jit(link_tensor, *centers, width = self.s4_l_width, axis = 1)
        self.sow("debug", f"link_tensor_sliced_{j}", link_tensor)  # Store the sliced link tensor for debugging
        x_result = link_multiply_real(link_tensor, x_partial)
        
        return x_result

    


# Inside class DysonNet:
    def get_config(self) -> Dict[str, Any]:
        config = {
            "token_size": self.token_size,
            "embedding_dim": self.embedding_dim,
            "n_blocks": self.n_blocks,
            "hidden_dim": self.hidden_dim,
            "seq_length": self.seq_length, # Used for s4_seq_length in factory
            "dropout_rate": self.dropout_rate,
            "use_gated_mlp": self.use_gated_mlp,
            "use_logcosh_output": self.use_logcosh_output,
            "logcosh_hidden": self.logcosh_hidden,
            "use_complex": self.use_complex,
            "complex_output": self.complex_output,
            "bidirectional": self.bidirectional,
            "use_convolution": self.use_convolution,
            "use_embedding_bias" : self.use_embedding_bias, 
            "embedding_type": self.embedding_type,
            "shift_project" : self.shift_project,
            "real_dtype": self.real_dtype,
            "complex_dtype": self.complex_dtype,
            # S4 parameters
            "s4_init_type": self.s4_init_type,
            "s4_decay_scale": self.s4_decay_scale,
            "s4_freq_scale": self.s4_freq_scale,
            "s4_long_range_frac": self.s4_long_range_frac,
            "s4_l_width": self.s4_l_width,                 # <<< ADDED
            "s4_include_interblock": self.s4_include_interblock, # <<< ADDED
            "partial_evaluation": self.partial_evaluation, # <<< ADDED
            "use_layer_norm_mixer": self.use_layer_norm_mixer,  # Include layer
            "use_symmetric_conv": self.use_symmetric_conv,  # Include symmetric convolution configuration
        }
        
        # Include any configuration dictionaries if they were provided/processed
        # These are the full configs as used by the DysonNet instance
        if hasattr(self, 's4_config') and self.s4_config is not None:
            config["s4_config_full"] = dict(unfreeze(self.s4_config)) # Store under a different key
        return config

    def __hash__(self):
        return hash((
            self.token_size,
            self.embedding_dim,
            self.n_blocks,
            self.hidden_dim,
            self.s4_config,          # FrozenDict, already hashable
            self.seq_length,
            self.dropout_rate,
            self.use_gated_mlp,
            self.use_logcosh_output,
            self.logcosh_hidden,
            self.use_complex,
            self.complex_output,
            self.bidirectional,
            self.use_convolution,
            self.embedding_type,
            self.s4_init_type,
            self.s4_decay_scale,
            self.s4_freq_scale,
            self.s4_long_range_frac,
            self.s4_include_interblock,
            self.partial_evaluation,
            self.s4_l_width,
            self.real_dtype,
            self.complex_dtype,
        ))




class DysonNetFactory:
    """
    Factory class to create DysonNet models with customizable configurations.
    
    This class provides a clean interface for creating DysonNet models with various
    configurations, making it easy to experiment with different architectures.
    """
    @staticmethod
    def create_model(
        token_size: int = 2,
        embedding_dim: int = 12,
        embedding_type: str = "linear",
        n_blocks: int = 3,
        hidden_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
        use_gated_mlp: bool = True,
        conv_kernel_size: int = 4,
        conv_layer_number : int = 1,
        
        # S4-specific parameters
        s4_seq_length: int = 128,
        s4_use_gating: bool = False,
        s4_config: Optional[Dict[str, Any]] = None,
        s4_states : int = 14,
        s4_init_type: str = "hippo",  # Options: "hippo", "random"
        s4_decay_scale: float = 1.0,  # For random initialization
        s4_freq_scale: float = 1.0,  # For random initialization
        s4_long_range_frac: float = 0.5,  # For random initialization
        s4_l_width: int = 128,
        s4_use_circulant_slice : bool = True,  # Whether to use circulant slice for S4 blocks
        s4_normalize : bool = False, 

        # Common parameters
        use_logcosh_output: bool = True,
        logcosh_hidden: int = 8,
        use_complex: bool = False,
        complex_output : bool = False,
        real_dtype: Any = jnp.float32,
        complex_dtype: Any = jnp.complex64,
        bidirectional: bool = True,
        use_convolution: bool = True,
        use_layer_norm_mixer : bool = False, 
        use_symmetric_conv: bool = False,  # Whether to use symmetric convolution in S4 blocks
        link_tensor_approximate : bool = False,
        shift_project : bool = False, 
        use_embedding_bias : bool = True, 
        marshall_rule : bool = False, 

        use_power_law : bool = False,
        power_law_modes : int = 1,
        power_init_range : float = 2.0,
        include_short_range : bool = False,

        s4_include_interblock: bool = False,
        partial_evaluation: bool = False
    ) -> DysonNet:
        """
        Create a DysonNet model with the specified configuration.
        :param s4_include_interblock: Whether to include inter-block S4 mean evaluation
        Args:
            token_size: Size of input tokens
            embedding_dim: Dimension for token embeddings
            n_blocks: Number of blocks to stack
            hidden_dim: Dimension for hidden representations
            dropout_rate: Dropout rate for regularization
            use_gated_mlp: Whether to use GatedMLP for feed-forward blocks
            conv_kernel_size: Kernel size for convolutional layers
            
            # S4-specific parameters
            s4_states: Number of state vectors for S4 blocks
            s4_seq_length: Maximum sequence length for S4 blocks
            s4_use_gating: Whether to use gating in S4 blocks
            s4_config: Additional configuration for S4 blocks
            
            # Common parameters
            use_logcosh_output: Fixed to True (LogCosh output is always used)
            logcosh_hidden: Number of hidden neurons for LogCosh layer
            use_complex: Whether to use complex-valued parameters in LogCosh
            bidirectional: Whether to use bidirectional processing
            use_convolution: Whether to use convolution layers
            
        Returns:
            Configured DysonNet model instance
        """
        
        # Build S4 configuration
        s4_base_config = {
            "d_state": s4_states,
            "l_max": s4_seq_length,
            "l_width" : s4_l_width,
            "use_conv": use_convolution,
            "conv_layer_number": conv_layer_number,
            "conv_kernel_size": conv_kernel_size,
            "use_gating": s4_use_gating,
            "bidirectional": bidirectional,
            "complex_params": use_complex,
            "dropout_rate": dropout_rate,
            "init_type": s4_init_type,
            "decay_scale": s4_decay_scale,
            "freq_scale": s4_freq_scale,
            "long_range_frac": s4_long_range_frac,
            "use_circulant_slice": s4_use_circulant_slice,
            "use_layer_norm_mixer": use_layer_norm_mixer,
            "use_symmetric_conv" : use_symmetric_conv,
            "link_tensor_approximate" : link_tensor_approximate, 
            "use_power_law": use_power_law,  # Whether to use power law initialization
            "power_law_modes": power_law_modes,  # Number of modes for power
            "power_init_range": power_init_range,  # Range for power law initialization
            "include_short_range": include_short_range,  # Whether to include short-range interactions
            "s4_normalize" : s4_normalize,
        }
        
        # Merge with any additional S4 configuration
        if s4_config:
            s4_base_config.update(s4_config)
        
        # Freeze configuration
        s4_base_config = freeze(s4_base_config)
        
        # Create and return the model
        return DysonNet(
             token_size=token_size,
             embedding_dim=embedding_dim,
             embedding_type=embedding_type,
             n_blocks=n_blocks,
             hidden_dim=hidden_dim,
             s4_config=s4_base_config,
             conv_layer_number=conv_layer_number,
             dropout_rate=dropout_rate,
             use_gated_mlp=use_gated_mlp,
             use_logcosh_output=use_logcosh_output,
             logcosh_hidden=logcosh_hidden,
             use_complex=use_complex,
             bidirectional=bidirectional,
             seq_length = s4_seq_length,
             use_convolution=use_convolution, 
             complex_output=complex_output,
             use_embedding_bias=use_embedding_bias,
             shift_project=shift_project,
             marshall_rule=marshall_rule, 
             real_dtype=real_dtype,
             complex_dtype=complex_dtype,
            # S4
            s4_init_type=s4_init_type,
            s4_decay_scale=s4_decay_scale,
            s4_freq_scale=s4_freq_scale,
            s4_long_range_frac=s4_long_range_frac,
            s4_l_width=s4_l_width,
            use_power_law=use_power_law,
            power_law_modes=power_law_modes,
            power_init_range=power_init_range,
            include_short_range=include_short_range,
            s4_include_interblock=s4_include_interblock,
            partial_evaluation=partial_evaluation, 
            link_tensor_approximate=link_tensor_approximate,
        )
