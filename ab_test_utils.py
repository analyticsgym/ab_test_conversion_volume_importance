"""Utility functions for common A/B test power and sample size calculations."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.stats import norm
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

# This code defines the list of public objects provided by this module, related to utility functions for A/B test power analysis, sample size calculations, and conversion rate statistics.
__all__ = [
    "ab_test_sample_requirements",
    "pct_change_funnel_level",
    "calc_ab_test_stats",
    "ab_test_sample_size_approx",
    "se_proportion",
]


def ab_test_sample_requirements(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    conversions_per_day: Optional[float] = None,
    traffic_per_day: Optional[float] = None,
) -> Dict[str, float]:
    """Calculate required visitors and conversions per group for a binary-outcome A/B test.

    Exactly one of ``conversions_per_day`` or ``traffic_per_day`` must be provided.

    Parameters
    ----------
    baseline_rate:
        Baseline conversion rate (e.g., 0.10 for 10%).
    mde:
        Minimum detectable relative effect (e.g., 0.05 for +5%).
    alpha:
        Significance level (default 0.05).
    power:
        Desired statistical power (default 0.80).
    conversions_per_day:
        Number of conversions observed per day across both variants.
    traffic_per_day:
        Number of visitors per day across both variants.
    """
    if (conversions_per_day is None) == (traffic_per_day is None):
        raise ValueError(
            "Provide exactly one of conversions_per_day or traffic_per_day."
        )

    variant_rate = baseline_rate * (1 + mde)
    effect_size = proportion_effectsize(baseline_rate, variant_rate)

    analysis = NormalIndPower()
    required_visitors = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        alternative="two-sided",
    )

    result: Dict[str, float] = {
        "baseline_rate": round(float(baseline_rate), 4),
        "mde": round(float(mde), 4),
        "alpha": round(float(alpha), 4),
        "power": round(float(power), 4),
        "num_groups": 2,
        "by_group_required_sample_size": required_visitors,
        "by_group_required_conversions_proxy": required_visitors * baseline_rate,
    }

    if traffic_per_day is not None:
        result["baseline_traffic_per_day"] = float(traffic_per_day)
        total_required_visitors = required_visitors * 2
        result["estimated_runtime_days_traffic"] = int(
            np.ceil(total_required_visitors / traffic_per_day)
        )
    else:
        result["baseline_conversions_per_day"] = float(conversions_per_day)
        total_required_conversions = (required_visitors * baseline_rate) * 2
        result["estimated_runtime_days_conversions"] = int(
            np.ceil(total_required_conversions / conversions_per_day)
        )

    return result


def pct_change_funnel_level(value: float, baseline_value: float) -> float:
    """Return percent change of ``value`` relative to ``baseline_value``."""
    if baseline_value == 0:
        raise ValueError("baseline_value must be non-zero to compute percent change.")
    return (value - baseline_value) / baseline_value


def calc_ab_test_stats(
    p1: float,
    p2: float,
    n1: float,
    n2: float,
    alpha: float = 0.05,
    verbose: bool = False,
) -> Dict[str, float]:
    """Compute two-sided p-value and theoretical power for an observed test."""
    z_obs = (p2 - p1) / np.sqrt((p1 * (1 - p1)) / n1 + (p2 * (1 - p2)) / n2)
    p_value = 2 * (1 - norm.cdf(abs(z_obs)))

    z_true = abs(z_obs)
    z_crit = norm.ppf(1 - alpha / 2)
    power_observed = 1 - norm.cdf(z_crit - z_true)

    conversions1 = n1 * p1
    conversions2 = n2 * p2

    if verbose:
        print(f"Group 1: rate={p1}, n={n1}, conversions={conversions1}")
        print(f"Group 2: rate={p2}, n={n2}, conversions={conversions2}")
        print(f"z={z_obs}, p-value={p_value}, power={power_observed}")

    return {
        "baseline_rate": p1,
        "variant_rate": p2,
        "group1_sample_size": n1,
        "group2_sample_size": n2,
        "group1_conversions": conversions1,
        "group2_conversions": conversions2,
        "z": z_obs,
        "p_value": p_value,
        "power": power_observed,
    }


def ab_test_sample_size_approx(baseline_rate: float, mde: float) -> float:
    """Approximate per-group sample size for an A/B test (alpha=0.05, power=0.8)."""
    delta = baseline_rate * mde
    return 16 * baseline_rate * (1 - baseline_rate) / (delta**2)


def se_proportion(p: float, n: float) -> float:
    """Standard error of a proportion given success rate ``p`` and sample size ``n``."""
    if n <= 0:
        raise ValueError("Sample size n must be positive.")
    return np.sqrt((p * (1 - p)) / n)
