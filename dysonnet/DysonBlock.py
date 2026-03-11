import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn
from typing import Any, Callable, Optional, Tuple

from dysonnet.S4 import *
from dysonnet.link_tensors import link_multiply_real
from dysonnet.partial_evaluation import (
    ActivationDifference,
    ActivationSum,
    difference_partial,
    slice_array_jit,
    sum_partial,
    unwrap,
)

def pad_to_length(x: jnp.ndarray, axis: int, target_length: int):
    """
    Pad `x` along `axis` with zeros at the end so its size becomes `target_length`.
    Returns:
      padded: the zero-padded array
      orig_len: the original length along that axis
      pad_before: number of zeros added before (here 0)
    """
    shape = x.shape
    orig_len = shape[axis]
    pad_total = target_length - orig_len
    if pad_total < 0:
        raise ValueError(f"target_length ({target_length}) < original length ({orig_len})")
    pad_before = 0
    pad_after  = pad_total

    # build pad_width: (before, after) for each dim
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (pad_before, pad_after)

    padded = jnp.pad(x, pad_width, mode='constant', constant_values=0)
    return padded, orig_len, pad_before

def remove_padding(padded: jnp.ndarray, axis: int, orig_len: int, pad_before: int = 0):
    """
    Remove zero-padding along `axis` that was added at the front (pad_before)
    and end so that the length along that axis goes back to `orig_len`.
    """
    # build slice object for each axis
    idx = [slice(None)] * padded.ndim
    idx[axis] = slice(pad_before, pad_before + orig_len)
    return padded[tuple(idx)]



# === your symmetric conv from before ===
class SymmetricDepthwiseConv1D(nn.Module):
    features: int
    kernel_size: int
    use_bias: bool = True
    dtype: Any = jnp.float32
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init:   Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        half = (self.kernel_size + 1) // 2
        kernel_half = self.param(
            'kernel_half', self.kernel_init,
            (half, 1, self.features), self.dtype
        )
        # reflect to get full kernel
        if self.kernel_size % 2 == 1:
            left = kernel_half[:-1]
        else:
            left = kernel_half
        right = jnp.flip(left, axis=0)
        kernel = jnp.concatenate([kernel_half, right], axis=0)  # [K,1,features]

        # simulate circular padding by wrapping the input
        pad_total = self.kernel_size - 1
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        x_padded = jnp.pad(x, [(0, 0), (pad_before, pad_after), (0, 0)], mode='wrap')
        y = jax.lax.conv_general_dilated(
            x_padded,
            kernel,
            window_strides=(1,),
            padding='VALID',
            dimension_numbers=('NWC','WIO','NWC'),
            feature_group_count=self.features,
        )
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,), self.dtype)
            y = y + bias
        return y




class DysonBlock(nn.Module):
    """
    S4 Block incorporating S4Kernel with optional gated features.

    Attributes:
        d_model: Dimension of the input and output features (d).
        l_max: Maximum sequence length for S4 kernel precomputation.
        d_state: State dimension (N) for the underlying S4 kernel. Default 64.
        use_conv: Whether to use a 1D depthwise convolution before S4. Default False.
        conv_kernel_size: Kernel size if use_conv is True. Default 4.
        conv_use_bias: Whether to use bias in convolution. Default True.
        use_gating: Whether to use gating (proj -> split -> S4/Gate -> combine). Default False.
        bidirectional: Whether to process sequence forward and backward with shared weights. Default False.
        complex_params: Whether the S4 kernel uses complex parameters and expects complex input
                        (split from real input of size d_model). Default True.
        dropout_rate: Dropout rate. Default 0.0.
        activation_fn: Activation function after convolution (if used). Default nn.silu.
        dtype: Data type for the module. Default jnp.float32.
        init_type: str = "hippo"  # Options: "hippo", "random"
        decay_scale: float = 1.0  # For random initialization
        freq_scale: float = 1.0  # For random initialization
        long_range_frac: float = 0.5  # For random initialization
        partial_evaluation : bool = False 
        l_width: Optional[int] = None, 
    
    """
    d_model: int
    l_max: int
    l_width: Optional[int] = None
    d_state: int = 64
    use_conv: bool = False
    conv_kernel_size: int = 4
    conv_use_bias: bool = True
    use_gating: bool = False
    bidirectional: bool = False
    complex_params: bool = True # Defaulting to complex as per S4 standard practice
    dropout_rate: float = 0.0
    activation_fn: Callable = nn.silu
    dtype: Any = jnp.float32
    complex_dtype: Any = None
    init_type: str = "hippo"  # Options: "hippo", "random"
    decay_scale: float = 1.0  # For random initialization
    freq_scale: float = 1.0  # For random initialization
    long_range_frac: float = 0.5  # For random initialization
    partial_evaluation : bool = False
    include_interblock : bool = False
    use_circulant_slice: bool = False # Use circulant matrix slices for S4 kernel
    block_index : int = 0
    use_layer_norm_mixer: bool = False # Whether to use LayerNormMixer for gating
    normalize_s4 : bool = False # Whether to normalize the S4 output (for debugging purposes)
    use_symmetric_conv: bool = False # Whether to use symmetric depthwise convolution
    link_tensor_approximate : bool = False
    conv_layer_number : int = 1 
    cache_dtype: Any = None # Default cache dtype for S4 kernel

    # Power law greens function 
    use_power_law : bool = False 
    power_law_modes : int = 1 
    power_init_range : float = 2.0# Range for power law initialization
    include_short_range : bool = False


    def _cache_dtype(self):
        return self.cache_dtype if self.cache_dtype is not None else self.dtype

    def setup(self):
        # Determine the dimension 'd' for the S4Kernel based on complex_params
        self.s4_kernel_dim = self.d_model // 2 if self.complex_params else self.d_model
        if self.complex_params and self.d_model % 2 != 0:
            raise ValueError("d_model must be even when complex_params=True")
        cache_dtype = self._cache_dtype()

        # Input/Output Projections for Gating
        self.in_proj_x = nn.Dense(
                features= self.d_model,
                use_bias=False,
                dtype=self.dtype,
                name="in_proj_x"
            )
        
        if self.use_gating:
            self.in_proj = nn.Dense(
                features= 2 * self.d_model,
                use_bias=False,
                dtype=self.dtype,
                name="in_proj"
            )

            self.out_proj = nn.Dense(
                features=self.d_model,
                use_bias=False, # Common SSM default, adjust if needed
                dtype=self.dtype,
                name="out_proj"
            )
        else:
            # Still use out_proj for consistency with gated blocks.
             self.out_proj = nn.Dense(
                features=self.d_model,
                use_bias=False,
                dtype=self.dtype,
                name="out_proj"
            )

        # Optional Convolution Layer (applied to the main path)
        if self.use_conv:
            
            conv_features = self.d_model # Conv operates on d_model before potential complex split
            
            if self.use_symmetric_conv:
                # Use symmetric depthwise convolution
                self.conv1d = SymmetricDepthwiseConv1D(
                    features=conv_features,
                    kernel_size=self.conv_kernel_size,
                    use_bias=self.conv_use_bias,
                    dtype=self.dtype,
                    name="conv1d"
                )
            else: 
                self.conv1d = nn.Conv(
                    features=conv_features,
                    kernel_size=(self.conv_kernel_size,),
                        # Depthwise convolution (per-channel)
                        feature_group_count=conv_features,
                        use_bias=self.conv_use_bias,
                        # Use 'SAME' for non-causal padding suitable for bidirectional/S4
                        padding='CIRCULAR',
                        dtype=self.dtype,
                        name="conv1d"
                    )

            if self.conv_layer_number > 1:
                conv_layers, conv_project = [], []
                activation_diff_conv_list, activation_sum_conv_list  = [], []
 
                for i in range(self.conv_layer_number - 1):
                    conv_layers.append(
                        nn.Conv(
                            features=conv_features,
                            kernel_size=(self.conv_kernel_size,),
                            # Depthwise convolution (per-channel)
                            feature_group_count=conv_features,
                            use_bias=self.conv_use_bias,
                            # Use 'SAME' for non-causal padding suitable for bidirectional/S4
                            padding='CIRCULAR',
                            dtype=self.dtype,
                            name=f"conv1d_{i+1}"
                        )
                    )
                    conv_project.append(nn.Dense(
                                        features= 2 * self.d_model,
                                        use_bias=False,
                                        dtype=self.dtype,
                                        name=f"conj_project_{i+1}"
                                    ))

                    if self.partial_evaluation:
                        activation_diff_conv_list.append(ActivationDifference(name=f"conv_diff_{i+1}", width=self.l_width, cache_dtype=cache_dtype))
                        activation_sum_conv_list.append(ActivationSum(name=f"conv_sum_{i+1}", width=self.l_width, cache_dtype=cache_dtype))

                self.conv_layers = tuple(conv_layers)
                self.conv_project = tuple(conv_project)
                self.activation_diff_conv_list = tuple(activation_diff_conv_list)
                self.activation_sum_conv_list = tuple(activation_sum_conv_list)

        # Instantiate the (vmapped) S4 Kernel
        # S4Kernel_vmap expects input shape (l, d_kernel) and outputs (l, d_kernel)
        # It's vmapped over the d_kernel dimension.
        self.s4_kernel = S4Kernel_vmap(
            d_state=self.d_state,
            l_max=self.l_max,
            l_width=self.l_width,
            complex_input=self.complex_params, # Kernel expects complex input if complex_params=True
            real_dtype=self.dtype,
            complex_dtype=self.complex_dtype,
            init_type=self.init_type,
            decay_scale=self.decay_scale,
            freq_scale=self.freq_scale,
            long_range_frac=self.long_range_frac,
            name="s4_kernel", 
            bidirectional = self.bidirectional, 
            use_circulant_slice = self.use_circulant_slice, # Use circulant matrix slices for S4 kernel
            normalize_s4 = self.normalize_s4, # Whether to normalize the S4 output (for debugging purposes)
            use_power_law = self.use_power_law,
            power_law_modes = self.power_law_modes,
            power_init_range = self.power_init_range,
            include_short_range = self.include_short_range,
            # d (feature dim for kernel) is implicitly handled by cloneLayer's vmap axes
        )

        # Dropout Layer
        self.dropout = nn.Dropout(rate=self.dropout_rate)

        # Diff layers for partial evaluation
        if self.partial_evaluation: 
            self.activation_diff_conv = ActivationDifference(name = "conv_diff", width = self.l_width, cache_dtype=cache_dtype)
            self.activation_sum_conv  = ActivationSum(name = "conv_sum", width = self.l_width, cache_dtype=cache_dtype) 

            #self.activation_diff_s4 = ActivationDifference(name = "s4_diff", width = self.l_width)
            self.activation_sum_s4  = ActivationSum(name = "s4_sum", width = self.l_width, cache_dtype=cache_dtype)
            self.activation_diff_g  = ActivationDifference(name = "g_diff", width = self.l_width, cache_dtype=cache_dtype)
            self.activation_diff_gating  = ActivationDifference(name = "gating_diff", width = self.l_width, cache_dtype=cache_dtype)


        if self.use_layer_norm_mixer:
            # LayerNormMixer for gating
            self.layer_norm_mixer = nn.LayerNorm(
                dtype=self.dtype,
                name="layer_norm_mixer", 
                use_bias= False, 
            )


    def __call__(self, x, g, centers = None, train: bool = True, mode: str = "full", link_buffer_mode : str = "",  link_buffer = None, interlayer_link_buffer = None):
        """
        Forward pass of the DysonBlock.

        Args:
            x: Input tensor of shape (batch, length, d_model).
            train: Whether the model is in training mode (for dropout).
            mode: Processing mode (e.g., "full").

        Returns:
            Output tensor of shape (batch, length, d_model).
        """


        batch_size, seq_len, d_model_in = x.shape
        assert d_model_in == self.d_model, "Input dimension mismatch"

        # Gating + Convolution 
        if self.use_gating:
            g, gate_val = self.apply_convolution(g, centers = centers, train=train, mode=mode) # Apply convolution and S4 kernel
        else: 
            g = self.apply_convolution(g, centers = centers, train=train, mode=mode)
        
        if self.conv_layer_number > 1:
            g = self.apply_deeper_convolution(g, centers = centers, train=train, mode=mode)

        if self.partial_evaluation and mode == "full":
            # Store g 
            self.sow("intermediates", "g_conv", g) # Store the gating output for debugging

        self.sow("debug", "before fuse", (x, g))
        u = self.fuse_with_x(x, g, mode=mode, centers=centers) # Fuse the input x with the gating output g
        if mode == "partial":
            self.sow("debug", "x_partial after fuse", u) # Store the partial input after fuse for debugging
            self.sow("debug", "g_partial after fuse", g) # Store the partial gating output for debugging


        # Propagate far field 
        if self.partial_evaluation and mode == "partial":
            assert centers is not None, "Centers must be provided for far field propagation"
            link_buffer, interlayer_link_buffer = self.update_link_buffers(u, link_buffer, interlayer_link_buffer, centers)

        self.sow("debug", "before s4", u)

        # # --- S4 Kernel Application ---
        y_s4 = self.apply_s4(u, train=train, mode=mode) # Apply S4 kernel
        self.sow("debug", "after s4", u)
        # Take care of the resumming for the difference propagation
        if self.partial_evaluation:
            y_s4 = self.activation_sum_s4(y_s4, centers = centers, mode = mode) 

        if self.partial_evaluation and mode == "partial" and link_buffer_mode == "interlayer" and self.block_index > 0:
            self.sow("debug", "interlayer_update", interlayer_link_buffer[self.block_index-1])
            y_s4 = y_s4 + interlayer_link_buffer[self.block_index-1]

        self.sow("debug", "after_sum", y_s4)

        if mode == "full" and self.partial_evaluation:
            # Store the S4 output for debugging
            self.sow("intermediates", "s4_output", y_s4)

        if mode == "full":
            self.sow("intermediates", "s4_output_full", y_s4)
        elif mode == "partial":
            self.sow("intermediates", "s4_output_partial", y_s4)

        # 7. Apply Dropout+
        if self.use_gating:
            y_final = self.apply_gating(y_s4, gate_val, mode = mode, train = train,centers= centers)

        else: 
            y_final = self.apply_gating(y_s4, mode = mode, train = train, centers= centers)

        # Apply the dropout layer
        g = self.dropout(g, deterministic=not train)
       

        if link_buffer_mode == "interlayer":
            self.sow("debug", "y_final", y_final) # Store the final output for debugging
            self.sow("debug", "link_buffer", link_buffer) # Store the far field mean for debugging
            self.sow("debug", "interlayer_link_buffer", interlayer_link_buffer) # Store the far field mean for debugging

            return y_final.astype(self.dtype), g.astype(self.dtype), link_buffer.astype(self.dtype), interlayer_link_buffer.astype(self.dtype)
        elif link_buffer_mode == "final": 
            self.sow("debug", "y_final", y_final) # Store the final output for debugging
            self.sow("debug", "link_buffer", link_buffer) # Store the far field mean for debugging

            return y_final.astype(self.dtype), g.astype(self.dtype), link_buffer.astype(self.dtype)
        else: 
            return y_final.astype(self.dtype), g.astype(self.dtype) # Explictly cast to block's dtype
    
    def forward_and_pullback(self, x, g, centers = None, train: bool = True, mode: str = "full", link_buffer_mode : bool = False,  prev_far_field = 0):
        """
        Forward pass of the DysonBlock.

        Args:
            x: Input tensor of shape (batch, length, d_model).
            train: Whether the model is in training mode (for dropout).
            mode: Processing mode (e.g., "full").

        Returns:
            Output tensor of shape (batch, length, d_model).
        """
        assert not link_buffer_mode, "link_buffer_mode is not supported in forward_and_pullback"
        assert mode == "full", "mode must be 'full' in forward_and_pullback"

        batch_size, seq_len, d_model_in = x.shape
        assert d_model_in == self.d_model, "Input dimension mismatch"

        # Gating + Convolution 
        if self.use_gating:
            g, gate_val = self.apply_convolution(g, centers = centers, train=train, mode=mode) # Apply convolution and S4 kernel
        else: 
            g = self.apply_convolution(g, centers = centers, train=train, mode=mode)
        
        if self.use_conv and self.conv_layer_number > 1:
            g = self.apply_deeper_convolution(g, centers = centers, train=train, mode=mode)

        self.sow("debug", "s4_block_input", x)

        u, pullback_b_s4_fn = jax.vjp(lambda xx : self.fuse_with_x(xx, g), x)
        pullback_b_s4 = lambda x, fn=pullback_b_s4_fn : unwrap(fn(x))


        # --- S4 Kernel Application ---
        y_s4, pullback_s4_fn = jax.vjp(lambda xx : self.apply_s4(xx, train=train, mode=mode), u) # Apply S4 kernel
        pullback_s4 = lambda x, fn=pullback_s4_fn : unwrap(fn(x))

        # Take care of the resumming for the difference propagation
        if self.partial_evaluation:
            y_s4 = self.activation_sum_s4(y_s4, centers = centers, mode = mode) 

        # 6. Apply Gating
        if self.use_gating:
            y_final, pullback_a_s4_fn = jax.vjp(lambda xx: self.apply_gating(xx, gate_val, mode = mode, train = train), y_s4)
        else: 
            y_final, pullback_a_s4_fn = jax.vjp(lambda xx: self.apply_gating(xx, mode = mode, train = train), y_s4)

        pullback_a_s4 = lambda x, fn = pullback_a_s4_fn : unwrap(fn(x))
        # Apply dropout to g 
        g = self.dropout(g, deterministic=not train)

        return y_final.astype(self.dtype), g.astype(self.dtype), (pullback_b_s4, pullback_s4, pullback_a_s4)

    def forward_partial(self, x_full, x_partial, g_full, g_partial, link_buffer, skip_buffer, centers, train: bool = False):
        """
        Forward pass of the DysonBlock.

        Args:
            x: Input tensor of shape (batch, length, d_model).
            train: Whether the model is in training mode (for dropout).
            mode: Processing mode (e.g., "full").

        Returns:
            Output tensor of shape (batch, length, d_model).
        """

        batch_size_full, seq_len, d_model_in = x_full.shape
        assert d_model_in == self.d_model, "Input dimension mismatch"

        if not self.use_gating:
            raise NotImplementedError("Partial evaluation is not implemented for non-gated DysonBlock")
        if self.use_conv and self.conv_layer_number > 1:
            raise AssertionError("Deeper convolution not supported in forward_partial")

        
        g_full, g_partial, gate_full, gate_partial = self.apply_convolution_partial(g_full, g_partial, centers) # Apply convolution and S4 kernel

        # Full path
        u_full, pullback_b_s4_fn = jax.vjp(lambda xx : self.fuse_with_x(xx, g_full), x_full)
        pullback_b_s4 = lambda x, fn=pullback_b_s4_fn : unwrap(fn(x))

        # Partial fuse  
        u_partial = self.fuse_with_x_partial(x_partial, g_partial, g_full, centers)
        self.sow("debug", "x_partial after fuse", u_partial) # Store the partial input after fuse for debugging
        self.sow("debug", "g_partial after fuse", g_partial) # Store the partial gating output for debugging
        # --- S4 Kernel Application ---
        y_s4_full, pullback_s4_fn = jax.vjp(lambda xx : self.apply_s4(xx, train=train, mode="full"), u_full) # Apply S4 kernel
        pullback_s4 = lambda x, fn=pullback_s4_fn : unwrap(fn(x)) 

        # Partial S4 application 
        #u_partial_diff = difference_partial(u_full, u_partial, centers, self.l_width) # Difference between full and partial input
        link_buffer = link_buffer.at[self.block_index].set(u_partial)
        skip_buffer_new = u_partial

        if self.link_tensor_approximate:
            skip_buffer = self.get_approximate_link_tensor(skip_buffer, centers, train=train, incoming=True)
            skip_buffer_new = self.get_approximate_link_tensor(skip_buffer_new, centers, gating = nn.silu(gate_full), train=train, incoming=False)

        y_s4_partial = self.apply_s4(u_partial, train=train, mode="partial") # Apply S4 kernel on partial input
        
        # Sum previous outputs and link buffers
        y_s4_partial = sum_partial(y_s4_full, y_s4_partial, centers, self.l_width)
        self.sow("debug", "skip_buffer", skip_buffer) # Store the partial S4 output for debugging
        y_s4_partial = y_s4_partial + skip_buffer

        # 6. Apply Gating
        y_final_full, pullback_a_s4_fn = jax.vjp(lambda xx: self.apply_gating(xx, gate_full, mode = "full", train = train), y_s4_full)
        pullback_a_s4 = lambda x, fn = pullback_a_s4_fn : unwrap(fn(x)) 

        # Apply gating on partial output 
        activated_gate_partial = difference_partial(nn.silu(gate_full), nn.silu(gate_partial), centers, self.l_width) 
        y_final_partial = self.apply_gating_partial(y_s4_partial, activated_gate_partial, train=train)

        # Apply dropout to g 
        g_full = self.dropout(g_full, deterministic=not train)
        g_partial = self.dropout(g_partial, deterministic=not train)

        return (y_final_full.astype(self.dtype), 
                y_final_partial.astype(self.dtype), 
                g_full.astype(self.dtype), 
                g_partial.astype(self.dtype), 
                link_buffer.astype(self.dtype), 
                skip_buffer_new.astype(self.dtype),
                (pullback_b_s4, pullback_s4, pullback_a_s4))




    def compute_moments(self, x, g, gate_val):
        """
        Forward pass of the DysonBlock.

        Args:
            x: Input tensor of shape (batch, length, d_model).
            train: Whether the model is in training mode (for dropout).
            mode: Processing mode (e.g., "full").

        Returns:
            Output tensor of shape (batch, length, d_model).
        """
        batch_size, seq_len, d_model_in = x.shape
        assert d_model_in == self.d_model, "Input dimension mismatch"

        u = self.fuse_with_x(x, g, mode="full") # Fuse the input x with the gating output g

        # # --- S4 Kernel Application ---
        y_s4 = self.apply_s4(u, train=False, mode="full") # Apply S4 kernel

        # 7. Apply gating
        if self.use_gating:
            y_final = self.apply_gating(y_s4, gate_val, mode = "full", train = False)
        else: 
            y_final = self.apply_gating(y_s4, mode = "full", train = False)

        return y_final.astype(self.dtype) # Explictly cast to block's dtype


    @jax.profiler.annotate_function
    def fuse_with_x(self, x, g, mode: str = "full", centers = None): 

        """
        Fuse the input x with the gating output g.
        Args:
            x: Input tensor of shape (batch, length, d_model).
            g: Gating output tensor of shape (batch, length, d_model).
        Returns:
            Fused tensor of shape (batch, length, d_model).
        """
        if self.use_layer_norm_mixer:
            # Apply LayerNormMixer to the gating output
            g = self.layer_norm_mixer(g)

        x = self.in_proj_x(x) # (b, l, d_model)
        if self.partial_evaluation:
            return x + self.activation_diff_g(g, mode=mode, centers=centers) # WARNING : Removing conv for debugging purposes
        return x + g
    

    def fuse_with_x_partial(self, x_partial, g_partial, g_full, centers): 

        """
        Fuse the input x with the gating output g.
        Args:
            x: Input tensor of shape (batch, length, d_model).
            g: Gating output tensor of shape (batch, length, d_model).
        Returns:
            Fused tensor of shape (batch, length, d_model).
        """
        if self.use_layer_norm_mixer:
            # Apply LayerNormMixer to the gating output
            g_partial = self.layer_norm_mixer(g_partial)
            g_full = self.layer_norm_mixer(g_full) 

        x_partial = self.in_proj_x(x_partial) # (b, l, d_model)
        return x_partial + difference_partial(g_full, g_partial, centers, self.l_width) # WARNING : Removing conv for debugging purposes

    @jax.profiler.annotate_function
    def apply_convolution(self, x, centers = None, train : bool = False, mode: str = "full"):

        # 1. Optional Gating Projection
        if self.use_gating:
            y = self.in_proj(x) # (b, l, 2 * d_model)
            x_main, gate_val = jnp.split(y, 2, axis=-1) # (b, l, d_model) each
        else:
            x_main = x # (b, l, d_model)
            gate_val = None # No gate value


        # 2. Optional Convolution on Main Path
        u = x_main # Input to S4 path starts here
        if self.use_conv:
            # Allow for difference propagation
            if self.partial_evaluation:
                u = self.activation_diff_conv(u, centers = centers, mode = mode)
                u = self.conv1d(u) # (b, l, d_model)
                u = self.activation_sum_conv(u, centers = centers, mode=mode) 

                if mode == "partial":
                    conv_bias = self.get_variable("params", "conv1d")["bias"] 
                    u = u - conv_bias # Remove extra bias 
            else: 
                u = self.conv1d(u) # (b, l, d_model)

            u = self.activation_fn(u) # Apply activation after conv


        if self.use_gating: 
            return u, gate_val # Return the main path and gate value 
        else: 
            return u
        
    @jax.profiler.annotate_function
    def apply_convolution_partial(self, g_full, g_partial, centers, train: bool = False, mode: str = "full"):

        # 1. Optional Gating Projection
        if self.use_gating:
            g_full = self.in_proj(g_full) # (b, l, 2 * d_model)
            g_full, gate_full = jnp.split(g_full, 2, axis=-1) # (b, l, d_model) each

            g_partial = self.in_proj(g_partial) # (b, l, 2 * d_model)
            g_partial, gate_partial = jnp.split(g_partial, 2, axis=-1)
        else:
            gate_full = None # No gate value
            gate_partial = None # No gate value


        if self.use_conv:
            
            g_partial = difference_partial(g_full, g_partial, centers, self.l_width)

            # Apply convolution 
            g_full = self.conv1d(g_full) # (b, l, d_model) 
            g_partial = self.conv1d(g_partial) # (b, l, d_model) 

            g_partial = sum_partial(g_full, g_partial, centers, self.l_width)
            conv_bias = self.get_variable("params", "conv1d")["bias"] 
            g_partial = g_partial - conv_bias # Remove extra bias
    
            g_full = self.activation_fn(g_full) # Apply activation after conv
            g_partial = self.activation_fn(g_partial) # Apply activation after conv

        if self.use_gating: 
            return g_full, g_partial, gate_full, gate_partial # Return the main path and gate value 
        else: 
            return g_full, g_partial
        
    def apply_deeper_convolution(self, x, centers = None, train: bool = True, mode: str = "full"):
        """
        Apply a series of convolutional layers to the input tensor.
        """

        for j in range(self.conv_layer_number-1):
            x = self.conv_project[j](x)
            if self.partial_evaluation:
                x = self.activation_diff_conv_list[j](x, centers = centers, mode = mode)
                x = self.conv_layers[j](x) # (b, l, d_model)
                x = self.activation_sum_conv_list[j](x, centers = centers, mode=mode)

                if mode == "partial":
                    conv_bias = self.get_variable("params", f"conv1d_{j+1}")["bias"]
                    x = x - conv_bias # Remove extra bias
            else: 
                x = self.conv_layers[j](x) # (b, l, d_model)

            x = self.activation_fn(x) # Apply activation after conv


        return x

    @jax.profiler.annotate_function
    def apply_s4(self, u, train: bool = True, mode: str = "full"):
        """
        Apply the S4 kernel to the input tensor.

        Args:
            x: Input tensor of shape (batch, length, d_model).
            train: Whether the model is in training mode (for dropout).
            mode: Processing mode (e.g., "full").

        Returns:
            Output tensor of shape (batch, length, d_model).
        """
        batch_size, seq_len, d_model_in = u.shape
        expected_kernel_dtype = self.dtype

        if self.complex_params:
            # Split d_model into real and imaginary parts for the kernel
            s4_input = u.reshape(batch_size, seq_len, 2, -1) # (b, l, 2, d_model/2)
        else:
            # Kernel expects real input
            s4_input = u # (b, l, d_model) float32

        # Pass mode to the underlying kernel call
        def apply_s4_kernel_single_batch(batch_item):
            batch_item = batch_item.astype(expected_kernel_dtype)
            # Pass mode to the kernel's __call__ method
            return self.s4_kernel(batch_item, mode == "partial", mode == "full") 

        # Vmap over the batch dimension (axis 0)
        #self.sow("debug", "s4_input", s4_input) # Store the S4 input for debugging
        s4_out_forward = jax.vmap(apply_s4_kernel_single_batch)(s4_input)
        # 4. Bidirectionality
        if self.bidirectional and mode == "partial" and not self.use_circulant_slice:
            # Flip input along sequence length for backward pass
            s4_out_forward = s4_out_forward # Shift for Hydra 
            s4_input_flipped = jnp.flip(s4_input, axis=1)
            # Apply the SAME kernel (weight sharing), passing mode
            s4_out_backward_flipped = jax.vmap(apply_s4_kernel_single_batch)(s4_input_flipped)
            s4_out_backward = jnp.flip(s4_out_backward_flipped, axis=1)
            s4_out = s4_out_forward + s4_out_backward
        else:
            s4_out = s4_out_forward

        # 5. Reconstruct Real Output if using Complex Params
        if self.complex_params:
            # s4_out is complex (b, l, 2, d_model/2)
            y_s4 = s4_out.reshape(batch_size, seq_len, self.d_model).astype(self.dtype) # (b, l, d_model)
        else:
            # s4_out is real (b, l, d_model)
            y_s4 = s4_out.astype(self.dtype) # Already (b, l, d_model) float

        #self.sow("validate", "s4_out", y_s4) # Store the S4 output for debugging
        return y_s4 



    @jax.profiler.annotate_function
    def apply_gating(self, y_s4, gate_val = 0, mode = "full", train = False, centers = None, use_stored_gating = False):
                # 6. Apply Output Projection and Optional Gating
        self.sow("debug", "y_final_before_gating", y_s4) # Store the S4 output before gating for debugging
        if self.use_gating:
            # Apply silu to the gate value
            if use_stored_gating: 
                activated_gate = self.activation_diff_gating.get_stored_activation(mode, centers)
            else: 
                activated_gate = nn.silu(gate_val)

                if self.partial_evaluation:
                    activated_gate = self.activation_diff_gating(activated_gate, mode = mode, centers = centers)

            out_projected = self.out_proj(y_s4)
            y_final = out_projected * activated_gate

        else:
            # Just apply the output projection
            y_final = self.out_proj(y_s4)

        self.sow("debug", "y_final", y_final) # Store the final output for debugging

        y_final = self.dropout(y_final, deterministic=not train)

        #self.sow("debug", "y_final_dropout", y_final) # Store the final output after dropout for debugging
        if mode == "full": 
            self.sow("intermediates", "activated_gate", self.dropout(activated_gate, deterministic=not train))
            self.sow("intermediates", "alt_gate", activated_gate)
        return y_final
    @jax.profiler.annotate_function
    def apply_gating_partial(self, y_s4_partial, activated_gate, train = False):
        # 6. Apply Output Projection and Optional Gating
        if self.use_gating:
            out_projected = self.out_proj(y_s4_partial)
            y_final = out_projected * activated_gate
        else:
            # Just apply the output projection
            y_final = self.out_proj(y_s4_partial)

        y_final = self.dropout(y_final, deterministic=not train)

        return y_final
        
    def sow_jacobian(self, jacobian, include_s4 : bool = True): 
        tag = "_before_s4" if include_s4 else "_after_s4"
        jacobian_cast = jacobian.astype(self._cache_dtype()) # Ensure the jacobian is in the correct dtype
        self.sow("intermediates", "jacobian" + tag, jacobian_cast) # Store the jacobian for debugging

    def sow_link_tensor(self, link_tensor):
        """
        Store the link tensor for debugging.
        Args:
            link_tensor: Link tensor to be stored.
        """
        link_tensor_cast = link_tensor.astype(self._cache_dtype()) # Ensure the link tensor is in the correct dtype
        self.sow("intermediates", "two_layer_link_tensor", link_tensor_cast)


    def update_link_buffers(self, s4_input, link_buffer, interlayer_link_buffer, centers, train : bool = False): 
        """
        Propagate the far field through the S4 block.
        Args:
            s4_input: Input tensor from the S4 kernel.
            link_buffer: Link buffer tensor.
            interlayer_link_buffer: Interlayer link buffer tensor.
        Returns:
            mean_exclude_near_field: Far field tensor for the pooling layer
        """ 
        assert centers is not None, "Centers must be provided for far field propagation"

        # In the first layer we need to propagate \Delta x = x - x0
        #if self.block_index == 0: 
        #    s4_input = self.activation_diff_s4(s4_input, centers = centers, mode = "partial") # Apply the activation difference
        self.sow("debug", "centers", centers)

        # Get the data
        jac_with_s4 = unwrap(self.get_variable("intermediates", "jacobian_before_s4")) # Shape (b, dp, l, d) with
        jac_with_s4 = slice_array_jit(jac_with_s4, *centers, width = self.l_width, axis = 1)
        self.sow("debug", "jac_with_s4_sliced", jac_with_s4)
        self.sow("debug", "jac input", s4_input) # Store the S4 input for debug
        link_buffer_update = jnp.einsum('bkld, bld -> bk', jac_with_s4, s4_input) 
        
        link_buffer = link_buffer + link_buffer_update

        if self.include_interblock and self.has_variable("intermediates", "two_layer_link_tensor"): 
            two_layer_link_tensor = unwrap(self.get_variable("intermediates", "two_layer_link_tensor")) 
            two_layer_link_tensor = slice_array_jit(two_layer_link_tensor, *centers, width = self.l_width, axis=1)
            self.sow("debug", "two_layer_link_tensor_sliced", two_layer_link_tensor)
            self.sow("debug", "link tensor input", s4_input) # Store the S4 input for debugging
            interlayer_update = link_multiply_real(two_layer_link_tensor, s4_input)
            interlayer_link_buffer = interlayer_link_buffer.at[self.block_index].add(interlayer_update) # Update the interlayer link buffer
        elif self.include_interblock and self.link_tensor_approximate: 

            if self.block_index > 0:
                delta = interlayer_link_buffer.at[self.block_index-1].get() # Get the previous interlayer link buffer value
                self.sow("debug", "approx_incoming_start", delta) # Store the interlayer link buffer for debugging
                delta = self.in_proj_x(delta)
                delta = self.apply_s4(delta, mode="partial", train=train)
                self.sow("debug", "approx_incoming_end", delta) # Store the interlayer link buffer after S4 for debugging
                interlayer_link_buffer = interlayer_link_buffer.at[self.block_index-1].set(delta) # Update the interlayer link buffer
            else: 
                # Write the data into the buffer 
                delta = s4_input
                self.sow("debug", "approx_outgoing_start", delta) # Store the interlayer link buffer for debugging

                # Apply s4 to the link tensors 
                delta = self.apply_s4(delta, mode="partial", train=train)

                # Apply the gating (vmapping over block index)
                delta = self.apply_gating(delta, 0, mode = "partial", train=train, centers = centers, use_stored_gating=True)
                self.sow("debug", "approx_outgoing_end", delta) # Store the interlayer link buffer after S4 for debugging
                interlayer_link_buffer = interlayer_link_buffer.at[self.block_index].add(delta) # Update the interlayer link buffer


        return link_buffer, interlayer_link_buffer


    def compute_link_tensor_approximateion(self, x, centers, mode): 
        
        if mode == "full":
            return 0 
        
        self.sow("debug", "link_correction_input", x)
        out_projected = self.out_proj(x)
        correction_gate = self.activation_diff_gating.get_stored_activation(mode, centers = centers)

        correction = correction_gate * out_projected 

        self.sow("debug", "link_correction_output", correction)


        return correction


    def get_approximate_link_tensor(self, delta, centers, train=False, incoming : bool = True, gating=None):
        if self.block_index > 0 and incoming: 
            self.sow("debug", "approx_incoming_start", delta)
            delta = self.in_proj_x(delta)
            delta = self.apply_s4(delta, mode="partial", train=train)
            self.sow("debug", "approx_incoming_end", delta)
        elif not incoming: 
            assert gating is not None, "Gating must be provided for incoming=False case"
            gating = slice_array_jit(gating, *centers, width=self.l_width) # Slice the gating tensor
            
            # Apply s4 to the link tensors
            self.sow("debug", "approx_outgoing_start", delta)
            delta = self.apply_s4(delta, mode="partial", train=train)

            # Apply the gating (vmappng over block index)
            delta = self.apply_gating_partial(delta, gating, train=train)
            self.sow("debug", "approx_outgoing_end", delta)
        
        return delta 
