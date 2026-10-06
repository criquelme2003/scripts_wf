import pytest

from matrix_construction import check_admissible, sparse_supercritical_block_matrix


def builds(n_N, n_M, c):
    try:
        sparse_supercritical_block_matrix(n_N, n_M, c, seed=0)
        return True
    except ValueError:
        return False


@pytest.mark.parametrize("N", [4, 20, 100])
@pytest.mark.parametrize("c_of", [
    lambda N: 0.125,
    lambda N: 1.0,
    lambda N: N / 2 - 1,      # borde admisible
    lambda N: N / 2 - 0.5,    # test2.py (c < N/2) lo deja pasar; aquí es inadmisible
    lambda N: N / 2,
    lambda N: float(N),
])
def test_check_admissible_matches_matrix_construction(N, c_of):
    c = c_of(N)
    n = N // 2
    assert (check_admissible(n, n, c) is None) == builds(n, n, c)


def test_check_admissible_accepts_border_and_rejects_fractional_above():
    assert check_admissible(50, 50, 49.0) is None
    reason = check_admissible(50, 50, 49.5)
    assert reason is not None and "p_M" in reason


@pytest.mark.parametrize("n_N, n_M", [(1, 0), (0, 1), (5, 1)])
def test_check_admissible_rejects_degenerate_sizes(n_N, n_M):
    assert check_admissible(n_N, n_M, 0.5) is not None
    assert not builds(n_N, n_M, 0.5)


def test_matrix_error_message_is_the_check_reason():
    with pytest.raises(ValueError) as exc:
        sparse_supercritical_block_matrix(50, 50, 49.5)
    assert str(exc.value) == check_admissible(50, 50, 49.5)
