def test_primes():
    from functools import reduce

    # The following algorithm is the Sieve of Eratosthenes:
    # https://stackoverflow.com/a/10640037

    N = 30

    assert reduce(
        (
            lambda r, x: [p for p in r if p not in list(range(x**2, N, x))]
            if (x in r)
            else r
        ),
        range(2, N),
        list(range(2, N)),
    ) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def test_deprecation():
    import warnings

    warnings.warn(
        "This is deprecated, but shouldn't raise an exception, unless "
        "enable_deprecations_as_exceptions() called from conftest.py",
        DeprecationWarning,
    )
