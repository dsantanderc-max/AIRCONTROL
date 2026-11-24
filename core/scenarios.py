from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from core.cyclone import CycloneDesignInputs, dimension_cyclone
from core.economics import (
    ControlAlternativeEconomics,
    compare_alternatives,
    evaluate_alternative,
)
from core.esp import ESPDesignInputs, dimension_esp
from core.gas_properties import DryGasComposition, summarize_gas_stream


@dataclass
class ScenarioConfig:
    name: str
    flow_cfm: float = 60000.0
    temp_c: float = 160.0
    pressure_atm: float = 1.0
    moisture_bws: float = 0.05
    viscosity_kg_ms: float = 2.42e-5
    particle_density_kg_m3: float = 2000.0
    inlet_conc_mg_m3: float = 1500.0
    standard_limit_mg_m3: float = 300.0
    o2_measured_pct: float = 8.0
    o2_ref_pct: float = 11.0
    use_cyclone_precontrol: bool = True
    use_esp: bool = True
    particle_size_dist: Optional[List[Tuple[float, float]]] = None
    economics: Optional[ControlAlternativeEconomics] = None


def combine_series_efficiencies(
    dist_df: pd.DataFrame,
    eta_1_pct: Optional[pd.Series] = None,
    eta_2_pct: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Calcula eficiencia fraccional y total en serie para 0, 1 o 2 etapas."""
    out = dist_df[["dp_um", "mass_pct"]].copy()
    f1 = (eta_1_pct / 100.0) if eta_1_pct is not None else pd.Series(0.0, index=out.index)
    f2 = (eta_2_pct / 100.0) if eta_2_pct is not None else pd.Series(0.0, index=out.index)

    eta_comb_frac = 1.0 - (1.0 - f1) * (1.0 - f2)
    out["eta_comb_pct"] = eta_comb_frac * 100.0
    out["contrib_comb_pct"] = out["eta_comb_pct"] * (out["mass_pct"] / 100.0)
    return out


def run_scenario(cfg: ScenarioConfig) -> Dict[str, object]:
    """Ejecuta un escenario completo de gas, precontrol (ciclón), PES y economía."""
    flow_m3_s = cfg.flow_cfm * 0.00047194745

    gas_summary = summarize_gas_stream(
        flow_m3_s=flow_m3_s,
        temp_c=cfg.temp_c,
        pressure_atm=cfg.pressure_atm,
        moisture_bws=cfg.moisture_bws,
        composition=DryGasComposition(o2_pct=cfg.o2_measured_pct),
        conc_mg_m3=cfg.inlet_conc_mg_m3,
        o2_ref_pct=cfg.o2_ref_pct,
        standard_limit_mg_m3=cfg.standard_limit_mg_m3,
    )

    cyclone_out = None
    esp_out = None
    eta_cyc_series = None
    eta_esp_series = None

    if cfg.use_cyclone_precontrol:
        cyc_inputs = CycloneDesignInputs(
            flow_m3_s=flow_m3_s,
            gas_density_kg_m3=float(gas_summary["rho_wet_kg_m3"] or 0.85),
            gas_viscosity_pa_s=cfg.viscosity_kg_ms,
            particle_density_kg_m3=cfg.particle_density_kg_m3,
            particle_size_dist=cfg.particle_size_dist,
        )
        cyclone_out = dimension_cyclone(cyc_inputs)
        eta_cyc_series = cyclone_out["table"]["eta_pct"]

    if cfg.use_esp:
        esp_inputs = ESPDesignInputs(
            flow_cfm=cfg.flow_cfm,
            temp_c=cfg.temp_c,
            viscosity_kg_ms=cfg.viscosity_kg_ms,
            moisture_pct=cfg.moisture_bws * 100.0,
            particle_size_dist=cfg.particle_size_dist,
        )
        esp_out = dimension_esp(esp_inputs)
        eta_esp_series = esp_out["table"]["eta_pct"]

    base_table = (
        esp_out["table"]
        if esp_out is not None
        else (
            cyclone_out["table"]
            if cyclone_out is not None
            else pd.DataFrame(
                cfg.particle_size_dist
                or [
                    (0.5, 20.0),
                    (1.0, 20.0),
                    (3.0, 10.0),
                    (5.0, 10.0),
                    (10.0, 20.0),
                    (20.0, 10.0),
                    (50.0, 10.0),
                ],
                columns=["dp_um", "mass_pct"],
            )
        )
    )

    combined_df = combine_series_efficiencies(
        dist_df=base_table,
        eta_1_pct=eta_cyc_series,
        eta_2_pct=eta_esp_series,
    )
    total_eff_pct = float(combined_df["contrib_comb_pct"].sum())
    outlet_conc_mg_m3 = cfg.inlet_conc_mg_m3 * (1.0 - total_eff_pct / 100.0)
    complies_standard = outlet_conc_mg_m3 <= cfg.standard_limit_mg_m3

    econ_summary = evaluate_alternative(cfg.economics) if cfg.economics is not None else None

    return {
        "scenario_name": cfg.name,
        "gas": gas_summary,
        "cyclone": cyclone_out,
        "esp": esp_out,
        "combined_table": combined_df,
        "total_efficiency_pct": total_eff_pct,
        "outlet_conc_mg_m3": outlet_conc_mg_m3,
        "standard_limit_mg_m3": cfg.standard_limit_mg_m3,
        "complies_standard": complies_standard,
        "economics": econ_summary,
    }


def default_scenarios() -> List[ScenarioConfig]:
    """Devuelve escenarios base para comparar en AIRCONTROL."""
    return [
        ScenarioConfig(
            name="PES Seco (Solo)",
            use_cyclone_precontrol=False,
            use_esp=True,
            economics=ControlAlternativeEconomics(
                name="PES Seco (Solo)",
                capex_usd=650000.0,
                annual_om_usd=25000.0,
                annual_energy_usd=8000.0,
                salvage_value_usd=15000.0,
                life_years=15,
                discount_rate_ea=0.12,
            ),
        ),
        ScenarioConfig(
            name="Ciclón + PES Seco (Serie)",
            use_cyclone_precontrol=True,
            use_esp=True,
            economics=ControlAlternativeEconomics(
                name="Ciclón + PES Seco (Serie)",
                capex_usd=730000.0,
                annual_om_usd=23000.0,
                annual_energy_usd=9500.0,
                salvage_value_usd=18000.0,
                life_years=15,
                discount_rate_ea=0.12,
            ),
        ),
    ]


def compare_scenarios(scenarios: List[ScenarioConfig]) -> pd.DataFrame:
    """Genera una tabla comparativa de desempeño técnico y económico."""
    records = []
    econ_list = [s.economics for s in scenarios if s.economics is not None]
    ranked_econ = {item["alternative"]: item for item in compare_alternatives(econ_list)}

    for sc in scenarios:
        res = run_scenario(sc)
        econ = ranked_econ.get(sc.name, {})
        records.append(
            {
                "Escenario": sc.name,
                "Eficiencia Total (%)": round(float(res["total_efficiency_pct"]), 2),
                "Conc. Salida (mg/m³)": round(float(res["outlet_conc_mg_m3"]), 2),
                "Límite (mg/m³)": sc.standard_limit_mg_m3,
                "Cumple Estándar": "Sí" if res["complies_standard"] else "No",
                "CAE (USD/año)": round(float(econ.get("equivalent_annual_cost_usd", 0.0)), 2),
            }
        )
    return pd.DataFrame(records)


if __name__ == "__main__":
    df_comp = compare_scenarios(default_scenarios())
    print("=== COMPARACIÓN DE ESCENARIOS ===")
    print(df_comp.to_string(index=False))