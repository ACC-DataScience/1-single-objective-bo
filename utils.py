import numpy as np
import torch


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def measure_yield_strength(
    time: float, temperature: float, v_prct: float, process: str
) -> float:
    """
    Synthetic objective function for estimating the yield strength of a precipitation
    strengthened alloy.

    Parameters:
    -----------
    time : float
        Aging time in hours (0.5-24).
    temperature : float
        Aging temperature in °C (500-1100).
    v_prct : float
        Vanadium percentage by weight (1-5%).
    process : str
        Pre-aging processing state: "RX" (recrystallization) or "CR" (cold rolling).

    Returns:
    --------
    float
        Measured yield strength of the alloy.
    """

    def gauss(x, mean, std):
        return np.exp(-(((x - mean) / std) ** 2))

    mean_v_eff = 20 * gauss(v_prct, 5, 2) - 15
    std_v_eff = 11 * (gauss(v_prct, 2.5, 2) + 0.05)

    time_eff = gauss(time, 7 - mean_v_eff, 8 - std_v_eff) + gauss(
        time, 3 - mean_v_eff, 2 - std_v_eff
    )

    temp_eff = (
        gauss(temperature, 800 - mean_v_eff, 100 - std_v_eff)
        + gauss(temperature, 750 - mean_v_eff, 150 - std_v_eff)
        + gauss(temperature, 500 - mean_v_eff, 250 - std_v_eff)
        + gauss(temperature, 750 - mean_v_eff, 800 - std_v_eff)
    )

    process_eff = {"RX": 0, "CR": 100}[process]

    return 100 * time_eff * temp_eff + process_eff + 300
