import numpy as np
from scipy.stats import norm

__all__ = [
    "two_afc",
    "duotrio_pc",
    "get_pguess",
    "pc2pd",
    "pd2pc",
    "discrim_2afc",
]

def two_afc(dprime: float) -> float:
    """Proportion correct in a 2-AFC task for a given d-prime."""
    return norm.cdf(dprime / np.sqrt(2))


def duotrio_pc(dprime: float) -> float:
    """Proportion correct in a duo-trio test for a given d-prime."""
    if dprime <= 0:
        return 0.5
    a = norm.cdf(dprime / np.sqrt(2.0))
    b = norm.cdf(dprime / np.sqrt(6.0))
    return 1 - a - b + 2 * a * b


def get_pguess(method: str = "duotrio", double: bool = False) -> float:
    """Return the chance performance level for a protocol."""
    method = method.lower()
    base = {
        "duotrio": 1 / 2,
        "twoafc": 1 / 2,
        "threeafc": 1 / 3,
        "triangle": 1 / 3,
        "tetrad": 1 / 3,
        "hexad": 1 / 10,
        "twofive": 1 / 10,
        "twofivef": 2 / 5,
    }.get(method)
    if base is None:
        raise ValueError(f"Unknown method: {method}")
    return base ** 2 if double else base


def pc2pd(pc: float, pguess: float) -> float:
    """Convert proportion correct to proportion discriminated."""
    if not (0 <= pguess <= 1):
        raise ValueError("pguess must be between 0 and 1")
    if not (0 <= pc <= 1):
        raise ValueError("pc must be between 0 and 1")
    pd = (pc - pguess) / (1 - pguess)
    return max(pd, 0.0)


def pd2pc(pd: float, pguess: float) -> float:
    """Convert proportion discriminated to proportion correct."""
    if not (0 <= pguess <= 1):
        raise ValueError("pguess must be between 0 and 1")
    if not (0 <= pd <= 1):
        raise ValueError("pd must be between 0 and 1")
    return pguess + pd * (1 - pguess)


def discrim_2afc(correct: int, total: int) -> dict:
    """Estimate d-prime from 2-AFC counts.

    Parameters
    ----------
    correct : int
        Number of correct responses.
    total : int
        Total number of trials.
    Returns
    -------
    dict
        Dictionary with keys ``d_prime`` and ``se``.
    """
    if total <= 0 or correct < 0 or correct > total:
        raise ValueError("invalid count data")
    pc = correct / total
    dprime = np.sqrt(2.0) * norm.ppf(pc)
    # derivative of pc->dprime mapping
    deriv = np.sqrt(2.0) / norm.pdf(norm.ppf(pc))
    se = np.sqrt(pc * (1 - pc) / total) * deriv
    return {"d_prime": dprime, "se": se}
