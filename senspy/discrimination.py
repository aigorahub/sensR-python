import numpy as np
from scipy.stats import norm, ncf
from scipy.integrate import quad

__all__ = [
    "two_afc",
    "duotrio_pc",
    "three_afc_pc",
    "triangle_pc",
    "tetrad_pc",
    "hexad_pc",
    "twofive_pc",
    "twofivef_pc",
    "discrim_2afc",
    "get_pguess",
    "pc2pd",
    "pd2pc",
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


def three_afc_pc(dprime: float) -> float:
    """Proportion correct for a 3-AFC protocol."""
    if dprime <= 0:
        return 1.0 / 3.0

    def integrand(x: float, delta: float) -> float:
        return norm.pdf(x - delta) * norm.cdf(x) ** 2

    val, _ = quad(integrand, -np.inf, np.inf, args=(dprime,))
    return float(np.clip(val, 1.0 / 3.0, 1.0))


def triangle_pc(dprime: float) -> float:
    """Proportion correct for the triangle test."""
    if dprime <= 0:
        return 1.0 / 3.0

    ncp = dprime ** 2 * (2.0 / 3.0)
    val = ncf.sf(3, 1, 1, ncp)
    return float(np.clip(val, 1.0 / 3.0, 1.0))


def tetrad_pc(dprime: float) -> float:
    """Proportion correct for the unspecified tetrad test."""
    if dprime <= 0:
        return 1.0 / 3.0

    def tetrad_fun(z: float, delta: float) -> float:
        return norm.pdf(z) * (
            2 * norm.cdf(z) * norm.cdf(z - delta) - norm.cdf(z - delta) ** 2
        )

    val, _ = quad(tetrad_fun, -np.inf, np.inf, args=(dprime,))
    pc = 1.0 - 2.0 * val
    return float(np.clip(pc, 1.0 / 3.0, 1.0))


def _poly_pc(dprime: float, coeffs: list[float], bounds: tuple[float, float], base: float) -> float:
    if dprime <= 0:
        return base
    low, high = bounds
    if dprime >= high:
        return 1.0
    powers = np.array([dprime ** i for i in range(len(coeffs))])
    val = float(np.dot(powers, coeffs))
    return float(np.clip(val, base, 1.0))


def hexad_pc(dprime: float) -> float:
    """Proportion correct for the hexad test."""
    coeffs = [
        0.0977646147,
        0.0319804414,
        0.0656128284,
        0.1454153496,
        -0.0994639381,
        0.0246960778,
        -0.0027806267,
        0.0001198169,
    ]
    return _poly_pc(dprime, coeffs, (0.0, 5.368), 0.1)


def twofive_pc(dprime: float) -> float:
    """Proportion correct for the two-out-of-five protocol."""
    coeffs = [
        0.0988496065454,
        0.0146108899965,
        0.0708075379445,
        0.0568876949069,
        -0.0424936635277,
        0.0114595626175,
        -0.0016573180506,
        0.0001372413489,
        -0.0000061598395,
        0.0000001166556,
    ]
    return _poly_pc(dprime, coeffs, (0.0, 9.28), 0.1)


def twofivef_pc(dprime: float) -> float:
    """Proportion correct for the two-out-of-five with forgiveness protocol."""
    coeffs = [
        0.399966014,
        0.001859461,
        0.194649607,
        0.021530254,
        -0.053426287,
        0.004419745,
        0.007685677,
        -0.003152163,
        0.000550084,
        -0.000047046,
        0.000001618,
    ]
    return _poly_pc(dprime, coeffs, (0.0, 4.333), 0.4)


def get_pguess(method: str, double: bool = False) -> float:
    """Guessing probability for a discrimination protocol."""
    mapping = {
        "duotrio": 0.5,
        "twoafc": 0.5,
        "threeafc": 1 / 3,
        "triangle": 1 / 3,
        "tetrad": 1 / 3,
        "hexad": 0.1,
        "twofive": 0.1,
        "twofivef": 2 / 5,
    }
    m = method.lower()
    if m not in mapping:
        raise ValueError(f"Unknown method: {method}")
    pg = mapping[m]
    return pg ** 2 if double else pg


def pc2pd(pc: float, pguess: float) -> float:
    """Convert proportion correct to proportion discriminated."""
    if not (0 <= pc <= 1 and 0 <= pguess <= 1):
        raise ValueError("values must be within [0, 1]")
    pd = (pc - pguess) / (1 - pguess)
    return float(np.clip(pd, 0.0, 1.0))


def pd2pc(pd: float, pguess: float) -> float:
    """Convert proportion discriminated to proportion correct."""
    if not (0 <= pd <= 1 and 0 <= pguess <= 1):
        raise ValueError("values must be within [0, 1]")
    pc = pguess + pd * (1 - pguess)
    return float(np.clip(pc, pguess, 1.0))


def discrim_2afc(correct: int, total: int) -> tuple[float, float]:
    """Estimate d-prime from 2-AFC data using a binomial approximation."""
    if total <= 0 or correct < 0 or correct > total:
        raise ValueError("invalid counts")

    pc = correct / total
    dp = norm.ppf(pc) * np.sqrt(2.0)
    se_pc = np.sqrt(pc * (1 - pc) / total)
    se_dp = se_pc / (norm.pdf(dp / np.sqrt(2.0)) / np.sqrt(2.0))
    return dp, se_dp
