import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Optional, Any
from flax.core import FrozenDict
import functools

DEFAULT_ACTIVATION = nn.silu


def fix_tuple(val, name):
    # --- FIX: Index into the tuple to get the actual array ---
    if not isinstance(val, tuple) or not val:
            raise TypeError(f"Expected a non-empty tuple for key '{name}', but got {type(val)}.")
    stored_activation = val[0]
    # ---------------------------------------------------------

    if not hasattr(stored_activation, 'shape'):
            raise TypeError(f"Value retrieved for key '{name}' is not an array-like object after tuple indexing. Got type: {type(stored_activation)}")

    return stored_activation

def slice_array(arr, idx, centers, *, width : int = 6, axis : int = 0):
    L = arr.shape[axis+1]  
    offsets = jnp.arange(-width, width+1)    # shape (K,)
    l_slice = (centers[:, None] + offsets) % L  # each row is the K positions

    #assert jnp.max(idx) == arr.shape[0] - 1, f"Index {jnp.max(idx)} out of bounds for array of shape {arr.shape}"

    @jax.vmap 
    def slice_one(i, l_slice):
        return jnp.take(arr[i], l_slice, axis = axis)
    
    return slice_one(idx, l_slice)

@functools.partial(jax.jit, static_argnames=('width','axis'))
def slice_array_jit(arr, idx, centers, *, width: int, axis: int = 0):
    return slice_array(arr, idx, centers, width=width, axis=axis)

def sum_partial(x_full, x_partial, centers, width):
    sliced_activation = slice_array_jit(x_full, *centers, width = width)

    if x_partial.shape != sliced_activation.shape: print(f"Warning: ActivationSum shape mismatch {x_partial.shape} vs {sliced_activation.shape}.")
    return x_partial + sliced_activation

def difference_partial(x_full, x_partial, centers, width):
    sliced_activation = slice_array_jit(x_full, *centers, width = width)

    if x_partial.shape != sliced_activation.shape: print(f"Warning: ActivationDifference shape mismatch {x_partial.shape} vs {sliced_activation.shape}.")
    return x_partial - sliced_activation




def slice_tokens(arr, idx, centers, *, width : int = 6, axis : int = 0, token_size: Optional[int] = 1):
    L = arr.shape[axis+1]  
    odd_shift = centers % token_size 
    centers = centers - odd_shift 
    offsets = jnp.arange(-token_size*width, token_size*width + token_size)    # shape (K,)
    l_slice = (centers[:, None] + offsets) % L  # each row is the K positions

    @jax.vmap 
    def slice_one(i, l_slice):
        return jnp.take(arr[i], l_slice, axis = axis)
    
    return slice_one(idx, l_slice) 


@functools.partial(jax.jit, static_argnames=('width', 'axis', 'token_size'))
def slice_tokens_jit(arr, idx, centers, *, width: int, axis: int = 0, token_size: Optional[int] = 1):
    return slice_tokens(arr, idx, centers, width=width, axis=axis, token_size=token_size)



class ActivationDifference(nn.Module):
    """
    Acts as identity in 'full' mode but sows input activation.
    In 'partial' mode, subtracts the corresponding stored activation
    from the input. Handles tuple wrapping from sow.
    """
    name: str  # Unique key to identify this activation point (e.g., "diff1/layer1_in")
    width : int
    cache_dtype: Any = jnp.float32  # Default cache dtype for stored activations


    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 mode: str,
                 centers = None 
                ) -> jnp.ndarray:
        if mode == 'full':
            if self.is_mutable_collection('intermediates'):
                # Sow the value - Flax might wrap it in a tuple internally
                self.sow('intermediates', self.name, x.astype(self.cache_dtype))
            return x



        elif mode == 'partial':

            # Retrieve the stored value (potentially a tuple)
            stored_value = self.get_variable('intermediates', self.name)
            stored_activation = fix_tuple(stored_value, self.name)
            stored_activation = slice_array_jit(stored_activation, *centers, width = self.width)

            self.sow("debug", "before diff", x.astype(self.cache_dtype))
            self.sow("debug", "after diff", x - stored_activation)
            self.sow("debug", "centers", centers)
            if x.shape != stored_activation.shape: print(f"Warning: ActivationDifference (key='{self.name}') shape mismatch {x.shape} vs {stored_activation.shape}.")
            return x - stored_activation
        else: raise ValueError(f"Unknown mode '{mode}' for ActivationDifference (key='{self.name}').")

    def get_stored_activation(self, mode, centers = None):
        """
        Retrieves the activation values for the current layer, either fully or partially sliced based on the mode.
        Args:
            mode (str): Determines the retrieval mode. If "full", returns the entire stored activation.
            centers (tuple or list): The center indices used for slicing the activation array when mode is not "full".
        Returns:
            numpy.ndarray: The activation values, either the full array or a sliced portion depending on the mode.
        Notes:
            - Uses `get_variable` to fetch stored intermediate activations.
            - Applies `fix_tuple` to ensure the activation is in the correct format.
            - If mode is not "full", slices the activation array around the provided centers with a specified width.
        """

        

        stored_value = self.get_variable('intermediates', self.name)
        stored_activation = fix_tuple(stored_value, self.name)
        
        if mode =="full":
            return stored_activation

        stored_activation = slice_array_jit(stored_activation, *centers, width = self.width)
        return stored_activation

class ActivationSum(nn.Module):
    """
    Acts as identity in 'full' mode but sows input activation.
    In 'partial' mode, adds the corresponding stored activation
    to the input. Handles tuple wrapping from sow.
    """
    name: str  # Unique key to identify this activation point (e.g., "sum1/layer1_out")
    width : int 
    cache_dtype: Any = jnp.float32  # Default cache dtype for stored activations

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 mode: str,
                 centers = None
                ) -> jnp.ndarray:
        if mode == 'full':
            if self.is_mutable_collection('intermediates'):
                self.sow('intermediates', self.name, x.astype(self.cache_dtype))
            return x
        elif mode == 'partial':
            stored_value = self.get_variable('intermediates', self.name)
            stored_activation = fix_tuple(stored_value, self.name)
            stored_activation = slice_array_jit(stored_activation, *centers, width = self.width)

            self.sow("debug", "before sum", x)
            self.sow("debug", "after sum", x + stored_activation)
            if x.shape != stored_activation.shape: print(f"Warning: ActivationSum (key='{self.name}') shape mismatch {x.shape} vs {stored_activation.shape}.")
            return x + stored_activation
        else: raise ValueError(f"Unknown mode '{mode}' for ActivationSum (key='{self.name}').")



class ActivationMean(nn.Module):
    """
    Acts as identity in 'full' mode but sows input activation.
    In 'partial' mode, adds the corresponding stored activation
    to the input. Handles tuple wrapping from sow.
    """
    name: str  # Unique key to identify this activation point (e.g., "sum1/layer1_out")

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 mode: str, 
                 centers = None
                ) -> jnp.ndarray:
        if mode == 'full':
            sum = jnp.sum(x, axis=1)
            if self.is_mutable_collection('intermediates'):
                self.sow('intermediates', self.name + "_sum", sum) 
            return jnp.zeros_like(sum) 
        elif mode == 'partial':
            # Retrieve stored activiations
            idx, _ = centers
            stored_mean = self.get_variable('intermediates', self.name + "_mean")
            stored_mean = fix_tuple(stored_mean, self.name + "_mean") 

            # Return stored value 
            return stored_mean[idx, ...]
        
        else: raise ValueError(f"Unknown mode '{mode}' for ActivationSum (key='{self.name}').")


class ActivationMeanOutside(nn.Module):
    """
    Acts as identity in 'full' mode but sows input activation.
    In 'partial' mode, adds the corresponding stored activation
    to the input. Handles tuple wrapping from sow.
    """
    name: str  # Unique key to identify this activation point (e.g., "sum1/layer1_out")
    L : int
    width : int 

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 mode: str, 
                 centers = None
                ) -> jnp.ndarray:
        if mode == 'full':
            sum = jnp.sum(x, axis=1)
            if self.is_mutable_collection('intermediates'):
                self.sow('intermediates', self.name, x)
                self.sow('intermediates', self.name + "_sum", sum) 
            return jnp.zeros_like(sum) 
        elif mode == 'partial':
            assert centers is not None, "centers must be provided in 'partial' mode for ActivationMeanOutside."

            # Retrieve stored activiations
            stored_x = unwrap( self.get_variable('intermediates', self.name) ) 
            stored_sum = unwrap( self.get_variable('intermediates', self.name + "_sum") )
            L0 = 2 * self.width + 1

            # Return stored value 
            idx, _ = centers 
            sum_inside = jnp.sum(slice_array_jit(stored_x, *centers, width = self.width), axis=1)
            mean_outside = (stored_sum[idx,...] - sum_inside) / (self.L - L0 )

            return mean_outside
        
        else: raise ValueError(f"Unknown mode '{mode}' for ActivationSum (key='{self.name}').")


def unwrap(x):
    return x[0]

def _remap_and_clip_single_array(array: jnp.ndarray, j: int, width: int, L : int, axis = None) -> jnp.ndarray:
    """
    Rolls a 3D array (batch, length, dim) so index j is centered, 
    then clips a region of size (2 * width + 1) around it.

    Args:
        array: The input 3D JAX ndarray.
        j: The index to center.
        width: The half-width of the clip region.

    Returns:
        The processed (rolled and clipped) JAX ndarray.

    Raises:
        ValueError: If input is not a 3D JAX array, j is out of bounds,
                    or the requested width is too large for the array length.
    """
    if not isinstance(array, jnp.ndarray) or array.ndim < 3:
        # Silently skip if not a 3D JAX array, or raise error?
        # Let's raise an error for clarity, handled by the caller.
        raise ValueError("Input must be a 3D JAX ndarray.")
        
    # Detect the correct axis 
    # Detect the correct axis (the one whose size equals L)
    if axis is None:
        matches = [i for i, dim in enumerate(array.shape) if dim == L]
        if not matches:
            raise ValueError(f"No axis found with length L={L} in array.shape {array.shape}")
        if len(matches) > 1:
            raise ValueError(
                f"Multiple axes {matches} match length L={L} in array.shape; "
                "please specify `axis` explicitly."
            )
        axis = matches[0]


    if axis != 1: 
        array = jnp.swapaxes(array, axis, 1)
    
    if not (0 <= j < L):
        print("input shape: ", array.shape)
        raise ValueError(f"Index j={j} is out of bounds for length l={L}.")
        
    expected_len = 2 * width + 1
    if expected_len <= 0:
        print("input shape: ", array.shape)
        raise ValueError(f"Width {width} must be non-negative.")
    if expected_len > L:
        print("input shape: ", array.shape)
        raise ValueError(f"Requested width {width} (total length {expected_len}) is larger than array length {l}.")

    # Calculate the center index of the array (integer division)
    center = L // 2
    # Calculate the shift needed to bring index j to the center
    shift = center - j
    
    # Roll the array along the length dimension (axis=1)
    rolled_array = jnp.roll(array, shift, axis=1)
    
    # Calculate the start and end indices for clipping around the new center
    start_idx = center - width
    end_idx = center + width + 1 # Python slicing is exclusive at the end
    
    # Clip the array
    clipped_array = rolled_array[:, start_idx:end_idx, ...]
    
    # Final check to ensure the output shape is as expected
    if clipped_array.shape[1] != expected_len:
         # This should not happen if previous checks and jnp.roll/slicing work correctly
         raise RuntimeError(f"Internal error: Clipped array shape {clipped_array.shape} does not match expected length {expected_len}")

    if axis != 1: 
        clipped_array = jnp.swapaxes(clipped_array, axis, 1)

    return clipped_array

def clip_activations_dict(data, j: int, width: int, L : int, print_warnings = True):
    """
    Recursively traverses a nested structure (dictionary, list, tuple) and 
    applies the _remap_and_clip_single_array function to any 3D JAX ndarray found.

    Args:
        data: The nested dictionary, list, or tuple containing activation arrays.
        j: The index to center around for clipping.
        width: The half-width of the clip region (total clipped length = 2*width + 1).
        
    Returns:
        A new nested structure with the same organization as the input, but with 
        all 3D JAX ndarrays processed (rolled and clipped). Other data types are 
        returned unchanged. If processing an array fails, it's returned unchanged.
    """
    if FrozenDict is not None and isinstance(data, FrozenDict):
        # Handle Flax FrozenDict specifically
        processed_items = {}
        for key, value in data.items():
            processed_items[key] = clip_activations_dict(value, j, width, L)
        return FrozenDict(processed_items)
        
    elif isinstance(data, dict):
        # Handle standard dictionaries
        processed_items = {}
        for key, value in data.items():
            processed_items[key] = clip_activations_dict(value, j, width, L)
        return processed_items
             
    elif isinstance(data, (list, tuple)):
         # Handle lists and tuples by processing each item
         return type(data)(clip_activations_dict(item, j, width, L) for item in data)

    elif isinstance(data, jnp.ndarray) and data.ndim >= 3 and data.shape[-1] != data.shape[-2]:
        # Apply the core logic only to 3D JAX arrays
        try:
            return _remap_and_clip_single_array(data, j, width, L)
        except ValueError as e:
            # Print a warning and return the original array if processing fails
            if print_warnings:
                print(f"Warning: Skipping array processing due to error: {e}. Returning original array.") 
            return data 
            
    else:
        # Return all other data types unchanged
        return data



def logcosh_head(x, use_complex: bool = False):
    """
    Computes the log-cosh activation function.
    """
    if use_complex:
        raise NotImplementedError("Complex logcosh is not implemented.")
    
    return jnp.sum(jnp.log(jnp.cosh(x)), axis= -1)
