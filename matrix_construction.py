import numpy as np

def check_admissible(n_N, n_M, c):
    """
    Returns None if (n_N, n_M, c) is admissible for
    sparse_supercritical_block_matrix, or the rejection reason otherwise.

    With n_N = n_M = N/2 this is equivalent to c <= N/2 - 1.
    """
    Ntot = n_N + n_M

    if Ntot <= 1:
        return "Ntot must be greater than 1."
    if n_M <= 1:
        return "n_M must be greater than 1."

    p_N = c / (Ntot - 1)
    p_M = c / (n_M - 1)

    if p_N > 1 or p_M > 1:
        return (
            f"Inadmissible Bernoulli parameter: "
            f"p_N={p_N:.4f}, p_M={p_M:.4f}, "
            f"for n_N={n_N}, n_M={n_M}, c={c}."
        )
    return None

def sparse_supercritical_block_matrix(n_N, n_M, c, seed=None):
    """
    Binary reflexive one-way block Bernoulli support matrix.

    Structure:
        E = [ E_NN  E_NM
              0     E_MM ]

    Here c controls the expected admissible out-degree in each block row:
        rows in N: p_N = c / (Ntot - 1)
        rows in M: p_M = c / (n_M - 1)

    Therefore:
        E[deg^+ | row in N] ≈ c
        E[deg^+ | row in M] ≈ c

    Cells with p_N > 1 or p_M > 1 are inadmissible and should be skipped,
    not truncated.
    """
    rng = np.random.default_rng(seed)

    reason = check_admissible(n_N, n_M, c)
    if reason is not None:
        raise ValueError(reason)

    Ntot = n_N + n_M
    p_N = c / (Ntot - 1)
    p_M = c / (n_M - 1)

    E = np.zeros((Ntot, Ntot), dtype=float)

    # N -> N and N -> M use p_N
    NN = rng.binomial(1, p_N, size=(n_N, n_N))
    NM = rng.binomial(1, p_N, size=(n_N, n_M))

    # M -> M uses p_M
    MM = rng.binomial(1, p_M, size=(n_M, n_M))

    # Remove self-loops before imposing reflexivity
    np.fill_diagonal(NN, 0)
    np.fill_diagonal(MM, 0)

    # Assemble one-way block matrix
    E[:n_N, :n_N] = NN
    E[:n_N, n_N:] = NM
    E[n_N:, :n_N] = 0
    E[n_N:, n_N:] = MM

    # Reflexivity
    np.fill_diagonal(E, 1)

    return E, p_N, p_M
