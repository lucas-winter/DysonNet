# Tests Folder

This folder holds pytest-based checks that target specific subsystems in the
DysonNet codebase (e.g., kernels, samplers, and operators). New tests should be added
here unless they are legacy or ad-hoc notebook experiments.

## How To Run

All tests in this folder:

```bash
/Users/lucas/.venvs/nqs/bin/python -m pytest tests -q
```

Single test file:

```bash
/Users/lucas/.venvs/nqs/bin/python -m pytest tests/<test_file.py> -q
```

If you want to see the test printouts (diagnostics), disable stdout capture:

```bash
/Users/lucas/.venvs/nqs/bin/python -m pytest tests/<test_file.py> -q -s
```

Fast mode (reduced matrix for difficult tests):

```bash
/Users/lucas/.venvs/nqs/bin/python -m pytest tests -q --fast
```

## Current Tests

`tests/test_local_update_kernel_tfim.py`

- What it tests: Verifies the fast local-update kernel for the TFIM operator
  matches the standard (exact) kernel evaluation.
- How: Builds small TFIM systems and compares fast vs standard kernel outputs
  across several sizes/tokenizations.
- Pass condition: `np.testing.assert_allclose` between fast and standard kernels.

`tests/test_typewriter_sampler_tfim.py`

- What it tests: Validates the typewriter sampler debug path for the TFIM setup
  (single-spin flips), including LUT warmup/burn-in and the debug mismatch
  checks against exact acceptance.
- How: Runs multiple `lut_multiply` values and multiple token sizes (L scales
  with token_size), then prints debug stats including total error, Metropolis
  approximation error, log-amplitude deviation, and screening (freeze) stats.
- Pass condition: `errors == 0` for the default `lut_multiply=1.5`.
- To see the printed diagnostics, run with `-s` as shown above.
