import jax
import jax.numpy as jnp


def check_cyclic_shift_invariance(
    layer_cls,
    N_state,
    L_seq,
    key,
    shifts_to_test=(1, 5),
    complex_input=False,
):
    """
    Tests if the sum of the layer output is invariant under cyclic input shifts.

    Args:
        layer_cls: The S4 layer class (e.g., S4BlockCorrected).
        N_state: The state dimension for the layer.
        L_seq: The sequence length (must match layer's l_max).
        key: JAX PRNG key.
        shifts_to_test: A tuple of integer shifts to apply.

    Returns:
        bool: True if the sum is invariant for all tested shifts, False otherwise.
    """
    print(f"Testing {layer_cls.__name__} with N_state={N_state}, L_seq={L_seq}")

    init_key, input_key, shift_key = jax.random.split(key, 3)

    # Instantiate the layer
    layer = layer_cls(d_state=N_state, l_max=L_seq, complex_input=complex_input)

    # Generate random complex input u
    # For NQS, inputs are often complex
    u_real = jax.random.normal(input_key, (L_seq, 16))
    u_imag = jax.random.normal(input_key, (L_seq, 16))  # Use same key is okay here
    u = u_real
    if complex_input:
        u = u_real + 1j * u_imag
    print(f"Input shape: {u.shape}, dtype: {u.dtype}")

    # Initialize layer parameters
    params = layer.init(init_key, u)["params"]
    print("Layer initialized successfully.")

    # --- Apply layer (use jax.jit for efficiency) ---
    @jax.jit
    def apply_layer(p, x):
        return layer.apply({"params": p}, x)

    # Get output for original input
    y = apply_layer(params, u)
    sum_y = jnp.sum(y)
    print(f"Original output sum: {sum_y}")

    all_invariant = True
    # Test different shifts
    for s in shifts_to_test:
        shift_amount = s % L_seq  # Ensure shift is within bounds
        if shift_amount == 0:
            continue  # Skip zero shift

        print(f"\n--- Testing shift s = {shift_amount} ---")

        # Create cyclically shifted input
        u_shifted = jnp.roll(u, shift=shift_amount, axis=0)

        # Get output for shifted input
        y_shifted = apply_layer(params, u_shifted)
        sum_y_shifted = jnp.sum(y_shifted)
        print(f"Shifted output sum:  {sum_y_shifted}")

        # Check if sums are close (allow for floating point inaccuracies)
        # Use complex-aware comparison if needed, but sum comparison is often sufficient
        is_close = jnp.allclose(sum_y, sum_y_shifted, rtol=1e-5, atol=1e-5)
        print(f"Sum invariant for shift {shift_amount}? {is_close}")

        if not is_close:
            print(f"WARN: Sum changed significantly for shift {shift_amount}!")
            print(f"Difference: {jnp.abs(sum_y - sum_y_shifted)}")
            all_invariant = False

    print("\n--- Test Summary ---")
    if all_invariant:
        print("PASSED: Sum of outputs appears invariant under tested cyclic shifts.")
    else:
        print("FAILED: Sum of outputs changed under tested cyclic shifts.")

    return all_invariant
