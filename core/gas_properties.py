from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

# Constantes físicas y de referencia
R_ATM_M3_KMOL_K: float = 0.082057366  # [atm * m³ / (kmol * K)]
R_J_MOL_K: float = 8.314462618  # [J / (mol * K)]
MMHG_PER_ATM: float = 760.0
IN_H2O_PER_ATM: float = 406.782  # Aproximación estándar de laboratorio/muestreo


def c_to_k(temp_c: float) -> float:
    return temp_c + 273.15


def c_to_f(temp_c: float) -> float:
    return temp_c * 9.0 / 5.0 + 32.0


def atm_to_pa(pressure_atm: float) -> float:
    return pressure_atm * 101325.0


@dataclass
class DryGasComposition:
    """Fracciones molares/volumétricas en base seca (en %; deben sumar 100%)."""
    co2_pct: float = 10.0
    o2_pct: float = 8.0
    co_pct: float = 0.0
    n2_pct: float = 82.0

    def validate(self) -> None:
        total = self.co2_pct + self.o2_pct + self.co_pct + self.n2_pct
        if not math.isclose(total, 100.0, rel_tol=1e-4, abs_tol=1e-3):
            raise ValueError(
                f"La composición seca debe sumar 100% (actual: {total:.4f}%)."
            )

    def dry_molecular_weight(self) -> float:
        """Peso molecular en base seca Md [kg/kmol o g/mol] (EPA Método 3)."""
        self.validate()
        return (
            0.44 * self.co2_pct
            + 0.32 * self.o2_pct
            + 0.28 * (self.co_pct + self.n2_pct)
        )


def wet_molecular_weight(dry_mw: float, moisture_frac_bws: float) -> float:
    """
    Peso molecular húmedo Ms:
    Ms = Md * (1 - Bws) + 18.015 * Bws
    donde Bws es la fracción de humedad (0 <= Bws < 1).
    """
    if not (0.0 <= moisture_frac_bws < 1.0):
        raise ValueError("La fracción de humedad Bws debe estar en [0, 1).")
    return dry_mw * (1.0 - moisture_frac_bws) + 18.015 * moisture_frac_bws


def gas_density_kg_m3(
    molecular_weight_kg_kmol: float,
    pressure_atm: float,
    temp_c: float,
) -> float:
    """Densidad del gas ideal: rho = (P * MW) / (R * T)."""
    temp_k = c_to_k(temp_c)
    if temp_k <= 0 or pressure_atm <= 0:
        raise ValueError("Presión y temperatura absoluta deben ser mayores que cero.")
    return (pressure_atm * molecular_weight_kg_kmol) / (R_ATM_M3_KMOL_K * temp_k)


def adjust_flow_to_reference(
    flow_actual_m3_s: float,
    temp_actual_c: float,
    pressure_actual_atm: float,
    temp_ref_c: float = 25.0,
    pressure_ref_atm: float = 1.0,
    moisture_actual_bws: float = 0.0,
    dry_basis: bool = False,
) -> float:
    """Ajusta un caudal volumétrico a condiciones de referencia (y opcionalmente base seca)."""
    t_act_k = c_to_k(temp_actual_c)
    t_ref_k = c_to_k(temp_ref_c)
    if t_act_k <= 0 or t_ref_k <= 0 or pressure_actual_atm <= 0 or pressure_ref_atm <= 0:
        raise ValueError("Temperaturas absolutas y presiones deben ser mayores que cero.")
    if not (0.0 <= moisture_actual_bws < 1.0):
        raise ValueError("Bws debe estar en [0, 1).")

    q_ref = flow_actual_m3_s * (t_ref_k / t_act_k) * (pressure_actual_atm / pressure_ref_atm)
    if dry_basis:
        q_ref *= (1.0 - moisture_actual_bws)
    return q_ref


def correct_concentration_o2(
    conc_measured_mg_nm3: float,
    o2_measured_pct: float,
    o2_ref_pct: float = 11.0,
) -> float:
    """
    Corrección de concentración por oxígeno de referencia:
    C_CR = C_medida * ((21 - O2_ref) / (21 - O2_medido))
    """
    if o2_measured_pct >= 21.0 or o2_ref_pct >= 21.0:
        raise ValueError("Los porcentajes de O2 deben ser menores a 21%.")
    if conc_measured_mg_nm3 < 0:
        raise ValueError("La concentración no puede ser negativa.")
    return conc_measured_mg_nm3 * ((21.0 - o2_ref_pct) / (21.0 - o2_measured_pct))


def pollutant_mass_flow_kg_h(
    conc_mg_m3: float,
    flow_m3_h: float,
) -> float:
    """Flujo / carga contaminante FC [kg/h] a partir de concentración [mg/m³] y caudal [m³/h]."""
    return (conc_mg_m3 * flow_m3_h) * 1.0e-6


def air_pollution_unit_uca(
    conc_corrected_mg_m3: float,
    standard_limit_mg_m3: float,
) -> float:
    """Unidad de Contaminación Atmosférica (UCA = C_CR / Estándar)."""
    if standard_limit_mg_m3 <= 0:
        raise ValueError("El estándar de emisión debe ser mayor que cero.")
    return conc_corrected_mg_m3 / standard_limit_mg_m3


def summarize_gas_stream(
    flow_m3_s: float,
    temp_c: float,
    pressure_atm: float = 1.0,
    moisture_bws: float = 0.05,
    composition: Optional[DryGasComposition] = None,
    conc_mg_m3: Optional[float] = None,
    o2_ref_pct: Optional[float] = None,
    standard_limit_mg_m3: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """Entrega un resumen técnico unificado de la corriente gaseosa y su emisión."""
    comp = composition if composition is not None else DryGasComposition()
    md = comp.dry_molecular_weight()
    ms = wet_molecular_weight(md, moisture_bws)
    rho_wet = gas_density_kg_m3(ms, pressure_atm, temp_c)
    rho_dry = gas_density_kg_m3(md, pressure_atm, temp_c)

    q_wet_ref_m3_s = adjust_flow_to_reference(
        flow_actual_m3_s=flow_m3_s,
        temp_actual_c=temp_c,
        pressure_actual_atm=pressure_atm,
        moisture_actual_bws=moisture_bws,
        dry_basis=False,
    )
    q_dry_ref_m3_s = adjust_flow_to_reference(
        flow_actual_m3_s=flow_m3_s,
        temp_actual_c=temp_c,
        pressure_actual_atm=pressure_atm,
        moisture_actual_bws=moisture_bws,
        dry_basis=True,
    )

    conc_corr: Optional[float] = None
    fc_kg_h: Optional[float] = None
    uca: Optional[float] = None

    if conc_mg_m3 is not None:
        conc_corr = (
            correct_concentration_o2(conc_mg_m3, comp.o2_pct, o2_ref_pct)
            if o2_ref_pct is not None
            else conc_mg_m3
        )
        fc_kg_h = pollutant_mass_flow_kg_h(conc_corr, q_dry_ref_m3_s * 3600.0)
        if standard_limit_mg_m3 is not None:
            uca = air_pollution_unit_uca(conc_corr, standard_limit_mg_m3)

    return {
        "temp_c": temp_c,
        "temp_f": c_to_f(temp_c),
        "temp_k": c_to_k(temp_c),
        "pressure_atm": pressure_atm,
        "moisture_bws": moisture_bws,
        "dry_mw_kg_kmol": md,
        "wet_mw_kg_kmol": ms,
        "rho_dry_kg_m3": rho_dry,
        "rho_wet_kg_m3": rho_wet,
        "q_actual_m3_s": flow_m3_s,
        "q_wet_ref_m3_s": q_wet_ref_m3_s,
        "q_dry_ref_m3_s": q_dry_ref_m3_s,
        "conc_measured_mg_m3": conc_mg_m3,
        "conc_corrected_mg_m3": conc_corr,
        "pollutant_load_kg_h": fc_kg_h,
        "uca": uca,
    }


if __name__ == "__main__":
    summary = summarize_gas_stream(
        flow_m3_s=28.3168,  # ~60,000 CFM
        temp_c=160.0,
        pressure_atm=1.0,
        moisture_bws=0.05,
        composition=DryGasComposition(co2_pct=11.0, o2_pct=7.5, co_pct=0.2, n2_pct=81.3),
        conc_mg_m3=450.0,
        o2_ref_pct=11.0,
        standard_limit_mg_m3=300.0,
    )
    for k, v in summary.items():
        print(f"{k}: {v}")