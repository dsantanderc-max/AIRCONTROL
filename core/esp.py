from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Constantes y rangos típicos de diseño (Tema 6A - Calidad del Aire)
EPSILON_0: float = 8.85e-12  # Permitividad del vacío [C / (V * m)]
FT_TO_M: float = 0.3048
IN_TO_FT: float = 1.0 / 12.0
IN_TO_M: float = 0.0254
UM_TO_M: float = 1.0e-6

TYPICAL_RANGES: Dict[str, Tuple[float, float, str]] = {
    "spacing_in": (8.0, 12.0, "pulg"),
    "height_ft": (30.0, 40.0, "pies"),
    "length_ft": (25.0, 30.0, "pies"),
    "voltage_kv": (40.0, 50.0, "kV"),
    "gas_velocity_fps": (1.0, 20.0, "pie/s"),
    "corona_power_w_per_1kcfm": (50.0, 500.0, "W / (1000 pie³/min)"),
    "corona_current_ua_per_ft2": (5.0, 70.0, "µA / pie²"),
    "fields": (4.0, 8.0, "campos"),
    "specific_area_ft2_per_1kcfm": (100.0, 800.0, "pie² / (1000 pie³/min)"),
    "charge_factor_p": (1.5, 2.4, "adimensional"),
}


@dataclass
class ESPDesignInputs:
    # Etapa 1: Características de la corriente gaseosa y del material particulado
    flow_cfm: float  # Caudal total Q_T [pie³/min]
    temp_c: float  # Temperatura del gas [°C]
    viscosity_kg_ms: float  # Viscosidad del gas mu_g [kg/(m*s)]
    moisture_pct: float = 5.0  # Humedad del gas [%]
    pressure_drop_inh2o: float = 5.0  # Caída de presión [pulg H2O]
    required_efficiency_pct: Optional[float] = None  # Eficiencia requerida [%]

    # Etapa 2–7: Parámetros geométricos y eléctricos
    specific_area_ft2_per_1kcfm: float = 400.0  # [pie² / (1000 pie³/min)]
    spacing_in: float = 10.0  # Separación entre placas S [pulg]
    height_ft: float = 30.0  # Altura inicial de placa H [pies]
    length_ft: float = 30.0  # Longitud inicial de placa L [pies]
    adjust_dimensions_to_int_plates: bool = True
    corona_power_w_per_1kcfm: float = 400.0  # [W / (1000 pie³/min)]
    corona_current_ua_per_ft2: float = 28.0  # [µA / pie²]
    charge_factor_p: float = 2.0  # Factor de carga p (3D/(D+2)) [-]

    # Distribución granulométrica: lista de (dp_um, mass_fraction_pct)
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


def dielectric_charge_factor(dielectric_constant_d: float) -> float:
    """Calcula el factor de carga p = 3D / (D + 2) para D entre 2 y 8."""
    if dielectric_constant_d <= -2.0:
        raise ValueError("La constante dieléctrica D debe ser mayor que -2.")
    return (3.0 * dielectric_constant_d) / (dielectric_constant_d + 2.0)


def cunningham_correction_factor(dp_um: float, temp_k: float) -> float:
    """Factor de corrección de Cunningham: Kc = 1 + (9.73e-3 * T_K^0.5) / dp_um."""
    if dp_um <= 0:
        raise ValueError("El diámetro de partícula dp debe ser mayor que cero.")
    if temp_k <= 0:
        raise ValueError("La temperatura absoluta T [K] debe ser mayor que cero.")
    return 1.0 + (9.73e-3 * math.sqrt(temp_k)) / dp_um


def migration_velocity_ms(
    dp_um: float,
    temp_k: float,
    viscosity_kg_ms: float,
    voltage_kv: float,
    spacing_in: float,
    charge_factor_p: float = 2.0,
    epsilon_0: float = EPSILON_0,
) -> Tuple[float, float, float]:
    """
    Calcula Kc, campo eléctrico Ec [V/m] y velocidad de migración W [m/s]:
    W = (p * eps_0 * Ec^2 * dp_m / (3 * mu_g)) * Kc
    """
    if viscosity_kg_ms <= 0:
        raise ValueError("La viscosidad del gas debe ser mayor que cero.")
    spacing_half_m = (spacing_in * IN_TO_M) / 2.0
    if spacing_half_m <= 0:
        raise ValueError("La separación entre placas debe ser mayor que cero.")

    voltage_v = voltage_kv * 1000.0
    electric_field_vm = voltage_v / spacing_half_m
    kc = cunningham_correction_factor(dp_um=dp_um, temp_k=temp_k)
    dp_m = dp_um * UM_TO_M

    w_ms = (
        charge_factor_p
        * epsilon_0
        * (electric_field_vm ** 2)
        * dp_m
        / (3.0 * viscosity_kg_ms)
    ) * kc
    return kc, electric_field_vm, w_ms


def deutsch_fractional_efficiency(
    w_ms: float,
    plate_area_ft2: float,
    plate_flow_cfs: float,
) -> float:
    """
    Ecuación de Deutsch-Anderson por sección/placa:
    eta_i = 1 - exp(-(W_fps * (A_placa / 2)) / (Q_placa / 2))
    """
    if plate_flow_cfs <= 0:
        raise ValueError("El caudal por sección de placa Q_placa debe ser mayor que cero.")
    w_fps = w_ms / FT_TO_M
    exponent = -(w_fps * (plate_area_ft2 / 2.0)) / (plate_flow_cfs / 2.0)
    return 1.0 - math.exp(exponent)


def check_range(value: float, low: float, high: float) -> str:
    """Devuelve 'OK' si value está dentro de [low, high], o indica fuera de rango."""
    if low <= value <= high:
        return f"OK ({low}–{high})"
    if value < low:
        return f"Bajo (< {low}; rango {low}–{high})"
    return f"Alto (> {high}; rango {low}–{high})"


def dimension_esp(inputs: ESPDesignInputs) -> Dict[str, object]:
    """Ejecuta las etapas 1 a 9 del dimensionamiento de un PES seco."""
    if inputs.flow_cfm <= 0:
        raise ValueError("El caudal total Q_T debe ser mayor que cero.")

    # Etapa 1: Temperaturas y distribución másica
    temp_k = inputs.temp_c + 273.15
    temp_f = inputs.temp_c * 9.0 / 5.0 + 32.0
    flow_cfs = inputs.flow_cfm / 60.0

    dist_df = pd.DataFrame(inputs.particle_size_dist, columns=["dp_um", "mass_pct"])
    total_mass_pct = float(dist_df["mass_pct"].sum())
    if not np.isclose(total_mass_pct, 100.0, atol=1e-3):
        raise ValueError(
            f"La suma de % en masa debe ser 100% (actual: {total_mass_pct:.4f}%)."
        )
    dist_df["mass_frac"] = dist_df["mass_pct"] / 100.0

    # Etapa 2: Área de colección requerida Ar [pie²]
    required_area_ft2 = inputs.flow_cfm * (inputs.specific_area_ft2_per_1kcfm / 1000.0)

    # Etapa 3: Número de placas Np y ajuste geométrico
    plate_area_initial_ft2 = inputs.height_ft * inputs.length_ft
    if plate_area_initial_ft2 <= 0:
        raise ValueError("Altura y longitud de placa deben ser mayores que cero.")

    raw_plates = required_area_ft2 / plate_area_initial_ft2
    n_plates = max(2, int(math.ceil(raw_plates)))

    if inputs.adjust_dimensions_to_int_plates:
        # Ajuste cuadrático de H y L para cumplir el área requerida con Np placas
        adjusted_side_ft = math.sqrt(required_area_ft2 / n_plates)
        height_adj_ft = adjusted_side_ft
        length_adj_ft = adjusted_side_ft
        plate_area_ft2 = height_adj_ft * length_adj_ft
    else:
        height_adj_ft = inputs.height_ft
        length_adj_ft = inputs.length_ft
        plate_area_ft2 = plate_area_initial_ft2

    # Etapa 4: Verificación de velocidad del gas Vg [pie/s]
    plate_flow_cfs = flow_cfs / (n_plates - 1)
    spacing_ft = inputs.spacing_in * IN_TO_FT
    gas_velocity_fps = plate_flow_cfs / ((spacing_ft / 2.0) * height_adj_ft)

    # Etapa 5: Potencia de la corona P [W]
    corona_power_w = inputs.corona_power_w_per_1kcfm * (
        (plate_flow_cfs / 2.0) * 60.0 / 1000.0
    )

    # Etapa 6: Corriente de la corona I [A]
    corona_current_a = (
        inputs.corona_current_ua_per_ft2 * 1.0e-6 * (plate_area_ft2 / 2.0)
    )

    # Etapa 7: Voltaje aplicado V [kV]
    voltage_v = corona_power_w / corona_current_a if corona_current_a > 0 else 0.0
    voltage_kv = voltage_v / 1000.0

    # Etapas 8 y 9: Velocidad de migración W, eficiencia fraccional y total
    rows = []
    for _, row in dist_df.iterrows():
        dp_um = float(row["dp_um"])
        mass_pct = float(row["mass_pct"])
        mass_frac = float(row["mass_frac"])

        kc, e_field_vm, w_ms = migration_velocity_ms(
            dp_um=dp_um,
            temp_k=temp_k,
            viscosity_kg_ms=inputs.viscosity_kg_ms,
            voltage_kv=voltage_kv,
            spacing_in=inputs.spacing_in,
            charge_factor_p=inputs.charge_factor_p,
        )
        eta_i = deutsch_fractional_efficiency(
            w_ms=w_ms,
            plate_area_ft2=plate_area_ft2,
            plate_flow_cfs=plate_flow_cfs,
        )
        contrib_pct = eta_i * mass_pct

        rows.append(
            {
                "dp_um": dp_um,
                "mass_pct": mass_pct,
                "mass_frac": mass_frac,
                "kc": kc,
                "electric_field_vm": e_field_vm,
                "w_ms": w_ms,
                "w_fps": w_ms / FT_TO_M,
                "eta_frac": eta_i,
                "eta_pct": eta_i * 100.0,
                "contrib_pct": contrib_pct,
            }
        )

    results_df = pd.DataFrame(rows)
    total_efficiency_pct = float(results_df["contrib_pct"].sum())
    total_area_installed_ft2 = n_plates * plate_area_ft2

    compliance_status: Optional[str] = None
    if inputs.required_efficiency_pct is not None:
        compliance_status = (
            "CUMPLE"
            if total_efficiency_pct >= inputs.required_efficiency_pct
            else "NO CUMPLE"
        )

    summary = {
        "temp_c": inputs.temp_c,
        "temp_f": temp_f,
        "temp_k": temp_k,
        "flow_cfm": inputs.flow_cfm,
        "flow_cfs": flow_cfs,
        "required_area_ft2": required_area_ft2,
        "installed_area_ft2": total_area_installed_ft2,
        "raw_plates": raw_plates,
        "n_plates": n_plates,
        "height_ft": height_adj_ft,
        "length_ft": length_adj_ft,
        "plate_area_ft2": plate_area_ft2,
        "plate_flow_cfs": plate_flow_cfs,
        "gas_velocity_fps": gas_velocity_fps,
        "velocity_check": check_range(gas_velocity_fps, 1.0, 20.0),
        "corona_power_w": corona_power_w,
        "corona_current_a": corona_current_a,
        "voltage_kv": voltage_kv,
        "voltage_check": check_range(voltage_kv, 40.0, 50.0),
        "total_efficiency_pct": total_efficiency_pct,
        "required_efficiency_pct": inputs.required_efficiency_pct,
        "compliance_status": compliance_status,
    }

    return {"summary": summary, "table": results_df}


def ejemplo_clase_6a() -> Dict[str, object]:
    """Preset con los parámetros del ejercicio de la presentación Tema 6A."""
    inputs = ESPDesignInputs(
        flow_cfm=60000.0,
        temp_c=160.0,
        viscosity_kg_ms=2.42e-5,
        moisture_pct=5.0,
        pressure_drop_inh2o=5.0,
        specific_area_ft2_per_1kcfm=400.0,
        spacing_in=10.0,
        height_ft=30.0,
        length_ft=30.0,
        adjust_dimensions_to_int_plates=True,
        corona_power_w_per_1kcfm=400.0,
        corona_current_ua_per_ft2=28.0,
        charge_factor_p=2.0,
    )
    return dimension_esp(inputs)


if __name__ == "__main__":
    out = ejemplo_clase_6a()
    s = out["summary"]
    print("=== RESUMEN DIMENSIONAMIENTO PES (EJEMPLO 6A) ===")
    for k, v in s.items():
        print(f"{k}: {v}")
    print("\n=== TABLA DE EFICIENCIA FRACCIONAL ===")
    print(out["table"].to_string(index=False))