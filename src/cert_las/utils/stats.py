import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - optional runtime dependency
    scipy_stats = None


def safe_prob(p):
    return np.clip(p, 1e-12, 1.0 - 1e-12)


def normal_cdf(x):
    x = np.asarray(x, dtype=np.float64)
    return np.vectorize(lambda v: 0.5 * math.erfc(-v / math.sqrt(2.0)), otypes=[np.float64])(x)


def normal_ppf(p):
    """Acklam inverse-normal approximation, vectorized over numpy arrays."""
    p = safe_prob(np.asarray(p, dtype=np.float64))
    a = np.array([
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ])
    b = np.array([
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ])
    c = np.array([
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ])
    d = np.array([
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ])
    plow = 0.02425
    phigh = 1.0 - plow
    x = np.empty_like(p)

    mask = p < plow
    if np.any(mask):
        q = np.sqrt(-2.0 * np.log(p[mask]))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        x[mask] = num / den

    mask = (p >= plow) & (p <= phigh)
    if np.any(mask):
        q = p[mask] - 0.5
        r = q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        x[mask] = num / den

    mask = p > phigh
    if np.any(mask):
        q = np.sqrt(-2.0 * np.log(1.0 - p[mask]))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        x[mask] = -(num / den)

    return x


def shifted_probability(p, radius: float):
    return normal_cdf(normal_ppf(p) - float(radius))


def exact_binom_sf_one_sided(n_pos: int, n_eff: int) -> Tuple[float, str]:
    """Return P[Bin(n_eff, 0.5) >= n_pos]."""
    if n_eff <= 0:
        return 1.0, "n_eff=0"
    if scipy_stats is not None:
        try:
            return float(scipy_stats.binomtest(n_pos, n_eff, 0.5, alternative="greater").pvalue), "scipy"
        except Exception:
            pass
    log2 = math.log(2.0)
    terms = [
        math.lgamma(n_eff + 1) - math.lgamma(k + 1) - math.lgamma(n_eff - k + 1) - n_eff * log2
        for k in range(n_pos, n_eff + 1)
    ]
    m = max(terms)
    p = math.exp(m) * sum(math.exp(t - m) for t in terms)
    return float(min(max(p, 0.0), 1.0)), "fallback"


def student_t_quantile(probability: float, df: int) -> float:
    if scipy_stats is not None:
        return float(scipy_stats.t.ppf(probability, df))
    # Normal fallback is close when df is reasonably large.
    return float(normal_ppf(probability))


def hoeffding_epsilon(alpha: float, n: int) -> float:
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if int(n) <= 0:
        raise ValueError("n must be positive")
    return math.sqrt(math.log(1.0 / float(alpha)) / (2.0 * float(n)))


@dataclass(frozen=True)
class OwnershipResult:
    wr: float
    rp: float
    zeta: float
    threshold: float
    epsilon: float
    t_alpha: float
    gamma: float
    verified: bool
    alpha: float
    num_noises: int
    num_images: int


def ownership_threshold(
    zeta: float,
    *,
    num_noises: int,
    num_images: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Closed-form WR threshold from the final Cert-LAS theorem.

    num_noises is M and num_images is N in the paper. The t-quantile is the
    one-sided (1-alpha) quantile with N-1 degrees of freedom.
    """
    if int(num_noises) <= 0:
        raise ValueError("num_noises must be positive")
    if int(num_images) <= 0:
        raise ValueError("num_images must be positive")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    zeta = float(np.clip(float(zeta), 0.0, 1.0))
    mn = float(int(num_noises) * int(num_images))
    df = max(1, int(num_images) - 1)
    t_alpha = student_t_quantile(1.0 - float(alpha), df)
    t2 = t_alpha * t_alpha
    gamma = (
        (2.0 * mn * zeta + t2) ** 2
        - 4.0
        * (mn + t2)
        * (mn * zeta * zeta - t2 * zeta + t2 * zeta * zeta)
    )
    gamma = max(float(gamma), 0.0)
    threshold = (2.0 * mn * zeta + t2 + math.sqrt(gamma)) / (2.0 * (mn + t2))
    return float(np.clip(threshold, 0.0, 1.0)), float(t_alpha), float(gamma)


def ownership_from_rates(
    wm_rates: Sequence[float],
    clean_rates: Sequence[float],
    *,
    num_images: int,
    alpha: float = 0.05,
    reference_probability: Optional[float] = None,
    zeta_num_samples: Optional[int] = None,
) -> OwnershipResult:
    wm_values = [float(v) for v in wm_rates]
    clean_values = [float(v) for v in clean_rates]
    if not wm_values:
        raise ValueError("wm_rates cannot be empty")
    if not clean_values and reference_probability is None:
        raise ValueError("clean_rates cannot be empty")
    if clean_values and len(wm_values) != len(clean_values):
        raise ValueError("wm_rates and clean_rates must have the same length")

    num_noises = len(wm_values)
    wr = mean(wm_values)
    rp = float(reference_probability) if reference_probability is not None else mean(clean_values)
    zeta_count = int(zeta_num_samples or num_noises)
    epsilon = hoeffding_epsilon(float(alpha), zeta_count)
    zeta = float(min(rp + epsilon, 1.0))
    threshold, t_alpha, gamma = ownership_threshold(
        zeta,
        num_noises=num_noises,
        num_images=int(num_images),
        alpha=float(alpha),
    )
    return OwnershipResult(
        wr=wr,
        rp=rp,
        zeta=zeta,
        threshold=threshold,
        epsilon=epsilon,
        t_alpha=t_alpha,
        gamma=gamma,
        verified=bool(wr > threshold),
        alpha=float(alpha),
        num_noises=int(num_noises),
        num_images=int(num_images),
    )


@dataclass(frozen=True)
class RadiusResult:
    pb_mean: float
    pc_mean: float
    pb_lower: float
    zeta: float
    threshold: float
    epsilon: float
    dkw_epsilon: float
    t_alpha: float
    gamma: float
    verified: bool
    radius: float
    alpha: float
    num_clean: int
    num_noises: int
    num_images: int
    num_thresholds: int


def _radius_lower_bound(
    thresholds: np.ndarray,
    tail_prob_lowers: np.ndarray,
    *,
    radius: float,
    smoothing_scale: float,
) -> float:
    total = 0.0
    previous = 0.0
    for threshold, probability in zip(thresholds, tail_prob_lowers):
        increment = float(threshold) - previous
        if increment > 0:
            total += increment * float(shifted_probability(float(probability), float(radius) / float(smoothing_scale)))
        previous = float(threshold)
    return float(np.clip(total, 0.0, 1.0))


def certified_radius_from_rates(
    wm_rates: Sequence[float],
    clean_rates: Sequence[float],
    *,
    num_images: int,
    num_clean: Optional[int] = None,
    alpha: float = 0.05,
    num_thresholds: int = 100,
    smoothing_scale: float = 1.0,
    reference_probability: Optional[float] = None,
) -> RadiusResult:
    """Certified radius using the final empirical-threshold Cert-LAS condition."""
    wm_values = np.asarray([float(v) for v in wm_rates], dtype=np.float64)
    clean_values = np.asarray([float(v) for v in clean_rates], dtype=np.float64)
    if wm_values.size == 0:
        raise ValueError("wm_rates cannot be empty")
    if clean_values.size == 0 and reference_probability is None:
        raise ValueError("clean_rates cannot be empty")
    if clean_values.size > 0 and wm_values.size != clean_values.size:
        raise ValueError("wm_rates and clean_rates must have the same length")
    if int(num_thresholds) <= 0:
        raise ValueError("num_thresholds must be positive")
    if float(smoothing_scale) <= 0:
        raise ValueError("smoothing_scale must be positive")

    num_noises = int(wm_values.size)
    pb_mean = mean(wm_values)
    pc_mean = float(reference_probability) if reference_probability is not None else mean(clean_values)
    count = int(num_clean or num_noises)
    epsilon = hoeffding_epsilon(float(alpha), count)
    zeta = float(min(pc_mean + epsilon, 1.0))
    threshold, t_alpha, gamma = ownership_threshold(
        zeta,
        num_noises=num_noises,
        num_images=int(num_images),
        alpha=float(alpha),
    )

    quantiles = np.linspace(0.0, 1.0, int(num_thresholds), dtype=np.float64)
    thresholds = np.unique(np.clip(np.quantile(wm_values, quantiles), 0.0, 1.0))
    dkw_epsilon = hoeffding_epsilon(float(alpha), num_noises)
    tail_prob_lowers = np.asarray(
        [max(float(np.mean(wm_values >= threshold)) - dkw_epsilon, 0.0) for threshold in thresholds],
        dtype=np.float64,
    )
    pb_lower = _radius_lower_bound(
        thresholds,
        tail_prob_lowers,
        radius=0.0,
        smoothing_scale=float(smoothing_scale),
    )

    if pb_lower <= threshold:
        radius = 0.0
    else:
        low, high = 0.0, 1.0
        while (
            _radius_lower_bound(thresholds, tail_prob_lowers, radius=high, smoothing_scale=float(smoothing_scale))
            > threshold
            and high < 1e6
        ):
            high *= 2.0
        for _ in range(80):
            mid = (low + high) / 2.0
            if _radius_lower_bound(thresholds, tail_prob_lowers, radius=mid, smoothing_scale=float(smoothing_scale)) > threshold:
                low = mid
            else:
                high = mid
        radius = low

    return RadiusResult(
        pb_mean=float(pb_mean),
        pc_mean=float(pc_mean),
        pb_lower=float(pb_lower),
        zeta=zeta,
        threshold=float(threshold),
        epsilon=float(epsilon),
        dkw_epsilon=float(dkw_epsilon),
        t_alpha=float(t_alpha),
        gamma=float(gamma),
        verified=bool(float(pb_mean) > float(threshold)),
        radius=float(radius),
        alpha=float(alpha),
        num_clean=int(count),
        num_noises=int(num_noises),
        num_images=int(num_images),
        num_thresholds=int(num_thresholds),
    )


def certified_radius_from_means(
    pb_mean: float,
    pc_mean: float,
    *,
    num_noises: int,
    num_images: int,
    num_clean: Optional[int] = None,
    alpha: float = 0.05,
    num_thresholds: int = 100,
    apply_hoeffding_to_pb: bool = True,
) -> RadiusResult:
    """Closed-form radius used by the Cert-LAS evaluation scripts.

    pb_mean is the watermarked-side detection probability. pc_mean is the clean-side
    probability. The clean side forms zeta with a Hoeffding upper bound; pb can
    optionally use a matching lower bound for conservative reporting.
    """
    if int(num_noises) <= 0:
        raise ValueError("num_noises must be positive")
    if int(num_images) <= 0:
        raise ValueError("num_images must be positive")
    count = int(num_clean or int(num_noises))
    if count <= 0:
        raise ValueError("num_clean must be positive")
    epsilon = hoeffding_epsilon(float(alpha), count)
    pb_lower = float(pb_mean) - epsilon if apply_hoeffding_to_pb else float(pb_mean)
    pb_lower = float(np.clip(pb_lower, 0.0, 1.0))
    zeta = float(min(float(pc_mean) + epsilon, 1.0))

    threshold, t_alpha, gamma = ownership_threshold(
        zeta,
        num_noises=int(num_noises),
        num_images=int(num_images),
        alpha=float(alpha),
    )

    radius = float(normal_ppf(pb_lower) - normal_ppf(threshold))
    radius = max(radius, 0.0)
    return RadiusResult(
        pb_mean=float(pb_mean),
        pc_mean=float(pc_mean),
        pb_lower=pb_lower,
        zeta=zeta,
        threshold=float(threshold),
        epsilon=float(epsilon),
        dkw_epsilon=0.0,
        t_alpha=float(t_alpha),
        gamma=float(gamma),
        verified=bool(float(pb_mean) > float(threshold)),
        radius=radius,
        alpha=float(alpha),
        num_clean=int(count),
        num_noises=int(num_noises),
        num_images=int(num_images),
        num_thresholds=int(num_thresholds),
    )


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("Cannot compute mean of empty values")
    return float(np.mean(vals))
