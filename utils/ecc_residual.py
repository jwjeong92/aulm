import math


def _validate_ecc_params(k, t, bit_error_prob):
    if int(k) <= 0:
        raise ValueError(f"k must be positive, got {k}.")
    if int(t) < 0:
        raise ValueError(f"t must be non-negative, got {t}.")
    if not 0.0 <= float(bit_error_prob) <= 1.0:
        raise ValueError(f"bit_error_prob must be in [0, 1], got {bit_error_prob}.")


def infer_bch_parity_bits(k, t):
    k = int(k)
    t = int(t)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")
    if t < 0:
        raise ValueError(f"t must be non-negative, got {t}.")
    log2_k = math.log2(k)
    if int(log2_k) != log2_k:
        raise ValueError(f"BCH parity inference expects k to be a power of two, got {k}.")
    return int((int(log2_k) + 1) * t)


def _log_binomial_pmf(k, errors, bit_error_prob):
    if errors < 0 or errors > k:
        return float("-inf")
    if bit_error_prob == 0.0:
        return 0.0 if errors == 0 else float("-inf")
    if bit_error_prob == 1.0:
        return 0.0 if errors == k else float("-inf")
    return (
        math.lgamma(k + 1)
        - math.lgamma(errors + 1)
        - math.lgamma(k - errors + 1)
        + errors * math.log(bit_error_prob)
        + (k - errors) * math.log1p(-bit_error_prob)
    )


def binomial_tail_prob(k, t, bit_error_prob):
    _validate_ecc_params(k, t, bit_error_prob)
    k = int(k)
    t = int(t)
    bit_error_prob = float(bit_error_prob)

    if t >= k:
        return 0.0

    tail = 0.0
    for errors in range(t + 1, k + 1):
        tail += math.exp(_log_binomial_pmf(k, errors, bit_error_prob))
    return float(min(max(tail, 0.0), 1.0))


def expected_uncorrected_bits(k, t, bit_error_prob):
    _validate_ecc_params(k, t, bit_error_prob)
    k = int(k)
    t = int(t)
    bit_error_prob = float(bit_error_prob)

    if t >= k:
        return 0.0

    expected = 0.0
    for errors in range(t + 1, k + 1):
        expected += errors * math.exp(_log_binomial_pmf(k, errors, bit_error_prob))
    return float(expected)


def residual_bit_error_prob(k, t, bit_error_prob):
    k = int(k)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")
    return float(expected_uncorrected_bits(k, t, bit_error_prob) / k)


def residual_profile_from_kt(k, t, bit_error_probs):
    residual = []
    uncorrectable = []
    for bit_error_prob in bit_error_probs:
        residual.append(float(residual_bit_error_prob(k, t, bit_error_prob)))
        uncorrectable.append(float(binomial_tail_prob(k, t, bit_error_prob)))
    return residual, uncorrectable
