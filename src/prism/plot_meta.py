"""Default colormaps, ranges and categorical legends per variable.

Ported from cloudnetpy's plotting/plot_meta.py so this viewer's defaults
match the official Cloudnet quicklooks instead of a flat viridis-everything
fallback. Keyed by bare variable name (not product-qualified), since the
same variable name means the same physical quantity across products.
"""
from __future__ import annotations

from dataclasses import dataclass

_COLORS = {
    "green": "#3cb371",
    "darkgreen": "#253A24",
    "lightgreen": "#70EB5D",
    "yellowgreen": "#C7FA3A",
    "yellow": "#FFE744",
    "orange": "#ffa500",
    "pink": "#B43757",
    "red": "#F57150",
    "shockred": "#E64A23",
    "seaweed": "#646F5E",
    "seaweed_roll": "#748269",
    "white": "#ffffff",
    "lightblue": "#6CFFEC",
    "blue": "#209FF3",
    "skyblue": "#CDF5F6",
    "darksky": "#76A9AB",
    "darkpurple": "#464AB9",
    "lightpurple": "#6A5ACD",
    "purple": "#BF9AFF",
    "darkgray": "#2f4f4f",
    "lightgray": "#ECECEC",
    "gray": "#d3d3d3",
    "lightbrown": "#CEBC89",
    "lightsteel": "#a0b0bb",
    "steelblue": "#4682b4",
    "black": "#000000",
    "grey": "#808080",
}


@dataclass(frozen=True)
class ContinuousMeta:
    cmap: str = "viridis"
    plot_range: tuple[float, float] | None = None
    log_scale: bool = False


# label starting with "_" is a real category but hidden from the legend
# (cloudnetpy convention for rare/internal states); we keep it in the color
# lookup but skip it when building displayed legend entries.
CATEGORICAL: dict[str, tuple[tuple[str, str], ...]] = {
    "target_classification": (
        ("_Clear sky", _COLORS["white"]),
        ("Droplets", _COLORS["lightblue"]),
        ("Drizzle or rain", _COLORS["blue"]),
        ("Drizzle & droplets", _COLORS["purple"]),
        ("Ice", _COLORS["lightsteel"]),
        ("Ice & droplets", _COLORS["darkpurple"]),
        ("Melting ice", _COLORS["orange"]),
        ("Melting & droplets", _COLORS["yellowgreen"]),
        ("Aerosols", _COLORS["lightbrown"]),
        ("Insects", _COLORS["shockred"]),
        ("Aerosols & insects", _COLORS["pink"]),
    ),
    "detection_status": (
        ("_Clear sky", _COLORS["white"]),
        ("Lidar only", _COLORS["yellow"]),
        ("Uncorrected atten.", _COLORS["seaweed_roll"]),
        ("Radar & lidar", _COLORS["green"]),
        ("_No radar but unknown atten.", _COLORS["purple"]),
        ("Radar only", _COLORS["lightgreen"]),
        ("_No radar but known atten.", _COLORS["orange"]),
        ("Corrected atten.", _COLORS["skyblue"]),
        ("Clutter", _COLORS["shockred"]),
        ("_Lidar molecular scattering", _COLORS["pink"]),
    ),
    "signal_source_status": (
        ("Clear sky", _COLORS["white"]),
        ("Radar & lidar", _COLORS["green"]),
        ("Radar only", _COLORS["lightsteel"]),
        ("Lidar only", _COLORS["yellow"]),
    ),
    "radar_attenuation_status": (
        ("_Clear sky", _COLORS["white"]),
        ("Negligible", _COLORS["green"]),
        ("Minor", _COLORS["lightgreen"]),
        ("Moderate", _COLORS["yellow"]),
        ("Severe", _COLORS["red"]),
        ("Unquantifiable", _COLORS["seaweed_roll"]),
        ("Undetected", _COLORS["skyblue"]),
    ),
    "iwc_retrieval_status": (
        ("_No ice", _COLORS["white"]),
        ("Reliable", _COLORS["green"]),
        ("Uncorrected", _COLORS["orange"]),
        ("Corrected", _COLORS["lightgreen"]),
        ("Ice from lidar", _COLORS["yellow"]),
        ("_Ice above rain", _COLORS["darksky"]),
        ("Clear above rain", _COLORS["skyblue"]),
        ("Positive temp.", _COLORS["seaweed"]),
    ),
    "ier_retrieval_status": (
        ("_No ice", _COLORS["white"]),
        ("Reliable", _COLORS["green"]),
        ("Uncorrected", _COLORS["orange"]),
        ("Corrected", _COLORS["lightgreen"]),
        ("Ice from lidar", _COLORS["yellow"]),
        ("_Ice above rain", _COLORS["darksky"]),
        ("Clear above rain", _COLORS["skyblue"]),
        ("Positive temp.", _COLORS["seaweed"]),
    ),
    "lwc_retrieval_status": (
        ("No liquid", _COLORS["white"]),
        ("Reliable", _COLORS["green"]),
        ("Adjusted", _COLORS["lightgreen"]),
        ("New pixel", _COLORS["yellow"]),
        ("Invalid LWP", _COLORS["seaweed_roll"]),
        ("_Invalid LWP2", _COLORS["shockred"]),
        ("_Measured rain", _COLORS["orange"]),
    ),
    "drizzle_retrieval_status": (
        ("_No drizzle", _COLORS["white"]),
        ("Reliable", _COLORS["green"]),
        ("Below melting", _COLORS["lightgreen"]),
        ("Unfeasible", _COLORS["red"]),
        ("Drizzle-free", _COLORS["orange"]),
        ("Rain", _COLORS["seaweed"]),
    ),
    "der_retrieval_status": (
        ("_Clear sky", _COLORS["white"]),
        ("Reliable", _COLORS["green"]),
        ("Mixed phase", _COLORS["lightgreen"]),
        ("Unfeasible", _COLORS["red"]),
        ("Surrounding ice", _COLORS["lightsteel"]),
    ),
}

CONTINUOUS: dict[str, ContinuousMeta] = {
    "Zh": ContinuousMeta(plot_range=(-40, 15)),
    "Z": ContinuousMeta(plot_range=(-40, 15)),  # categorize's own name for radar reflectivity
    "ldr": ContinuousMeta(plot_range=(-30, -5)),
    "width": ContinuousMeta(plot_range=(1e-2, 1e0), log_scale=True),
    "v": ContinuousMeta(cmap="RdBu_r", plot_range=(-4, 4)),
    "skewness": ContinuousMeta(cmap="RdBu_r", plot_range=(-1, 1)),
    "kurtosis": ContinuousMeta(plot_range=(1, 5)),
    "phi_cx": ContinuousMeta(cmap="RdBu_r", plot_range=(-2, 2)),
    "rho_cx": ContinuousMeta(plot_range=(1e-2, 1e0), log_scale=True),
    "v_sigma": ContinuousMeta(plot_range=(1e-2, 1e0), log_scale=True),
    "insect_prob": ContinuousMeta(plot_range=(0, 1)),
    "radar_liquid_atten": ContinuousMeta(plot_range=(0, 5)),
    "radar_gas_atten": ContinuousMeta(plot_range=(0, 5)),
    "radar_rain_atten": ContinuousMeta(plot_range=(0, 15)),
    "radar_melting_atten": ContinuousMeta(plot_range=(0, 5)),
    "iwc": ContinuousMeta(plot_range=(1e-7, 1e-3), log_scale=True),
    "iwc_error": ContinuousMeta(cmap="RdYlGn_r", plot_range=(0, 5)),
    "lwc": ContinuousMeta(cmap="Blues", plot_range=(1e-5, 1e-2), log_scale=True),
    "lwc_error": ContinuousMeta(cmap="RdYlGn_r", plot_range=(0, 2)),
    "epsilon": ContinuousMeta(cmap="inferno", plot_range=(1e-7, 1e-1), log_scale=True),
    "epsilon_error": ContinuousMeta(cmap="inferno", plot_range=(1e-7, 1e-1), log_scale=True),
    "beta": ContinuousMeta(plot_range=(1e-7, 1e-4), log_scale=True),
    "beta_raw": ContinuousMeta(plot_range=(1e-7, 1e-4), log_scale=True),
    "beta_smooth": ContinuousMeta(plot_range=(1e-7, 1e-4), log_scale=True),
    "depolarisation": ContinuousMeta(plot_range=(1e-3, 1), log_scale=True),
    "depolarisation_raw": ContinuousMeta(plot_range=(1e-3, 1), log_scale=True),
    "ier": ContinuousMeta(plot_range=(2e-5, 6e-5)),
    "ier_error": ContinuousMeta(cmap="RdYlGn_r", plot_range=(1e-5, 5e-5)),
    "Do": ContinuousMeta(plot_range=(1e-6, 1e-3), log_scale=True),
    "Do_error": ContinuousMeta(cmap="RdYlGn_r", plot_range=(0.1, 0.5)),
    "der": ContinuousMeta(cmap="coolwarm", plot_range=(1e-6, 1e-4), log_scale=True),
    "der_error": ContinuousMeta(cmap="coolwarm", plot_range=(1e-6, 1e-4), log_scale=True),
    "N_scaled": ContinuousMeta(plot_range=(1e6, 1e9), log_scale=True),
    "mu": ContinuousMeta(plot_range=(0, 10)),
    "S": ContinuousMeta(plot_range=(0, 25)),
    "S_error": ContinuousMeta(cmap="RdYlGn_r", plot_range=(0.1, 0.5)),
    "drizzle_N": ContinuousMeta(plot_range=(1e4, 1e9), log_scale=True),
    "drizzle_N_error": ContinuousMeta(cmap="RdYlGn_r", plot_range=(0.1, 0.5)),
    "drizzle_lwc": ContinuousMeta(plot_range=(1e-8, 1e-3), log_scale=True),
    "drizzle_lwc_error": ContinuousMeta(cmap="RdYlGn_r", plot_range=(0.3, 1)),
    "drizzle_lwf": ContinuousMeta(plot_range=(1e-8, 1e-5), log_scale=True),
    "drizzle_lwf_error": ContinuousMeta(cmap="RdYlGn_r", plot_range=(0.3, 1)),
    "v_drizzle": ContinuousMeta(cmap="RdBu_r", plot_range=(-2, 2)),
    "v_drizzle_error": ContinuousMeta(cmap="RdYlGn_r", plot_range=(0.3, 1)),
    "v_air": ContinuousMeta(cmap="RdBu_r", plot_range=(-2, 2)),
    "Tw": ContinuousMeta(cmap="RdBu_r", plot_range=(-50, 50)),
    "temperature": ContinuousMeta(cmap="RdBu_r", plot_range=(-50, 50)),
}
