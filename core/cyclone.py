from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PI_UM_TO_M: float = 1.0e-6

# Proporciones geométricas de referencia tipo Stairmand (alta eficiencia)
STAIRMAND_HE_RATIOS: Dict[str, float] = {
    "a_over_d": 0.50,   # Altura de entrada / diámetro del cuerpo
    "b_over_d": 0.20,   # Ancho de entrada / diámetro del cuerpo
    "s_over_d": 0.50,   # Profundidad del tubo de salida / diámetro
    "h_over_d": 4.00,   # Altura total del ciclón / diámetro
    "l_over_d": 1.50,   # Altura del cilindro / diámetro
    "d_e_over_d": 0.50, # Diámetro del ducto de salida de gas / diámetro
    "b_d_over_d": 0.375,# Diámetro de descarga de sólidos / diámetro
}


@dataclass
class CycloneDesignInputs:
    flow_m3_s: float  # Caudal volumétrico a condiciones de operación [m³/s]
    gas_density_kg_m3: float  # Densidad del gas [kg/m³]
    gas_viscosity_pa_s: float  # Viscosidad dinámica del gas [Pa*s = kg/(m*s)]
    particle_density_kg_m3: float  # Densidad real de la partícula [kg/m³]
    inlet_velocity_target_m_s: float = 15.0  # Velocidad de entrada objetivo [m/s]
    config_type: str = "Stairmand HE"
    particle_size_dist: Optional[List[Tuple[float, float]]] = None

    def __post_init__(self) -> None:
        if self.particle_size_dist is None:
            self.particle_size_dist = [
                (0.5, 20.0),
                (1.0, 20.0),
                (3.0, 10.0),
                (5.0, 10.0),
                (10.0, 20.0),
                (20.0, 10.0),
                (50.0, 10.0),
            ]


def cyclone_cut_diameter_um(
    diameter_m: float,
    inlet_velocity_m_s: float,
    gas_viscosity_pa_s: float,
    particle_density_kg_m3: float,
    gas_density_kg_m3: float,
    n_turns: float = 5.0,
    b_over_d: float = 0.20,
) -> float:
    """
    Diámetro de corte d50 [µm] aproximado (modelo tipo Lapple):
    d50 = sqrt((9 * mu * b) / (2 * pi * Ne * Vi * (rho_p - rho_g)))
    """
    delta_rho = particle_density_kg_m3 - gas_density_kg_m3
    if delta_rho <= 0:
        raise ValueError("La densidad de la partícula debe superar la densidad del gas.")
    if inlet_velocity_m_s <= 0 or gas_viscosity_pa_s <= 0 or diameter_m <= 0:
        raise ValueError("Velocidad, viscosidad y diámetro deben ser mayores que cero.")

    b_m = b_over_d * diameter_m
    d50_m = math.sqrt(
        (9.0 * gas_viscosity_pa_s * b_m)
        / (2.0 * math.pi * n_turns * inlet_velocity_m_s * delta_rho)
    )
    return d50_m / PI_UM_TO_M


def cyclone_pressure_drop_pa(
    gas_density_kg_m3: float,
    inlet_velocity_m_s: float,
    k_head: float = 8.0,
) -> float:
    """Caída de presión aproximada en el ciclón: Delta_P = 0.5 * K * rho_g * Vi^2 [Pa]."""
    return 0.5 * k_head * gas_density_kg_m3 * (inlet_velocity_m_s ** 2)


def lapple_fractional_efficiency(dp_um: float, d50_um: float) -> float:
    """Eficiencia fraccional curva Lapple: eta_i = 1 / (1 + (d50 / dp)^2)."""
    if dp_um <= 0 or d50_um <= 0:
        raise ValueError("dp_um y d50_um deben ser mayores que cero.")
    return 1.0 / (1.0 + ((d50_um / dp_um) ** 2))


def dimension_cyclone(inputs: CycloneDesignInputs) -> Dict[str, object]:
    """Calcula geometría preliminar, caída de presión y eficiencias del ciclón."""
    if inputs.flow_m3_s <= 0 or inputs.inlet_velocity_target_m_s <= 0:
        raise ValueError("Caudal y velocidad de entrada deben ser mayores que cero.")

    a_over_d = STAIRMAND_HE_RATIOS["a_over_d"]
    b_over_d = STAIRMAND_HE_RATIOS["b_over_d"]
    inlet_area_target_m2 = inputs.flow_m3_s / inputs.inlet_velocity_target_m_s

    # Q = Vi * (a * b) = Vi * (a_over_d * b_over_d * D^2)
    diameter_m = math.sqrt(inlet_area_target_m2 / (a_over_d * b_over_d))
    inlet_height_m = a_over_d * diameter_m
    inlet_width_m = b_over_d * diameter_m
    actual_inlet_velocity_m_s = inputs.flow_m3_s / (inlet_height_m * inlet_width_m)

    d50_um = cyclone_cut_diameter_um(
        diameter_m=diameter_m,
        inlet_velocity_m_s=actual_inlet_velocity_m_s,
        gas_viscosity_pa_s=inputs.gas_viscosity_pa_s,
        particle_density_kg_m3=inputs.particle_density_kg_m3,
        gas_density_kg_m3=inputs.gas_density_kg_m3,
        b_over_d=b_over_d,
    )
    delta_p_pa = cyclone_pressure_drop_pa(
        gas_density_kg_m3=inputs.gas_density_kg_m3,
        inlet_velocity_m_s=actual_inlet_velocity_m_s,
    )

    dist_df = pd.DataFrame(inputs.particle_size_dist, columns=["dp_um", "mass_pct"])
    total_mass_pct = float(dist_df["mass_pct"].sum())
    if not np.isclose(total_mass_pct, 100.0, atol=1e-3):
        raise ValueError(
            f"La suma de % en masa debe ser 100% (actual: {total_mass_pct:.4f}%)."
        )

    rows = []
    for _, r in dist_df.iterrows():
        dp_um = float(r["dp_um"])
        mass_pct = float(r["mass_pct"])
        eta_i = lapple_fractional_efficiency(dp_um=dp_um, d50_um=d50_um)
        contrib_pct = eta_i * mass_pct
        rows.append(
            {
                "dp_um": dp_um,
                "mass_pct": mass_pct,
                "eta_frac": eta_i,
                "eta_pct": eta_i * 100.0,
                "contrib_pct": contrib_pct,
            }
        )

    table_df = pd.DataFrame(rows)
    total_efficiency_pct = float(table_df["contrib_pct"].sum())

    summary = {
        "config_type": inputs.config_type,
        "flow_m3_s": inputs.flow_m3_s,
        "diameter_m": diameter_m,
        "inlet_height_m": inlet_height_m,
        "inlet_width_m": inlet_width_m,
        "total_height_m": STAIRMAND_HE_RATIOS["h_over_d"] * diameter_m,
        "inlet_velocity_m_s": actual_inlet_velocity_m_s,
        "d50_um": d50_um,
        "delta_p_pa": delta_p_pa,
        "total_efficiency_pct": total_efficiency_pct,
    }
    return {"summary": summary, "table": table_df}


if __name__ == "__main__":
    out = dimension_cyclone(
        CycloneDesignInputs(
            flow_m3_s=28.3168,
            gas_density_kg_m3=0.8288,
            gas_viscosity_pa_s=2.42e-5,
            particle_density_kg_m3=2000.0,
            inlet_velocity_target_m_s=15.0,
        )
    )
    for k, v in out["summary"].items():
        print(f"{k}: {v}")
    print("\n=== EFICIENCIA FRACCIONAL CICLÓN ===")
    print(out["table"].to_string(index=False))