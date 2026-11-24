from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List


def effective_annual_to_periodic_rate(ea_rate: float, periods_per_year: int = 12) -> float:
    """
    Convierte una tasa Efectiva Anual (EA) a tasa periódica vencida i:
    i = (1 + EA)^(1 / m) - 1
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year debe ser mayor que cero.")
    if ea_rate <= -1.0:
        raise ValueError("La tasa EA debe ser mayor que -100% (-1.0).")
    return (1.0 + ea_rate) ** (1.0 / periods_per_year) - 1.0


def periodic_to_effective_annual_rate(periodic_rate: float, periods_per_year: int = 12) -> float:
    """
    Convierte una tasa periódica vencida i a tasa Efectiva Anual (EA):
    EA = (1 + i)^m - 1
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year debe ser mayor que cero.")
    if periodic_rate <= -1.0:
        raise ValueError("La tasa periódica debe ser mayor que -100% (-1.0).")
    return (1.0 + periodic_rate) ** periods_per_year - 1.0


def inflation_adjusted_rate(real_rate: float, inflation_rate: float) -> float:
    """Tasa combinada / aparente ajustada por inflación (IPC): i_c = i_r + f + i_r * f."""
    return real_rate + inflation_rate + (real_rate * inflation_rate)


def factor_f_given_p(rate: float, n_periods: int) -> float:
    """Factor de cantidad compuesta de pago único (F/P, i, n) = (1 + i)^n."""
    if n_periods < 0:
        raise ValueError("n_periods no puede ser negativo.")
    return (1.0 + rate) ** n_periods


def factor_p_given_f(rate: float, n_periods: int) -> float:
    """Factor de valor presente de pago único (P/F, i, n) = (1 + i)^(-n)."""
    return 1.0 / factor_f_given_p(rate=rate, n_periods=n_periods)


def factor_f_given_a(rate: float, n_periods: int) -> float:
    """Factor de cantidad compuesta de serie uniforme (F/A, i, n)."""
    if n_periods <= 0:
        raise ValueError("n_periods debe ser mayor que cero.")
    if math.isclose(rate, 0.0, abs_tol=1e-12):
        return float(n_periods)
    return (((1.0 + rate) ** n_periods) - 1.0) / rate


def factor_a_given_f(rate: float, n_periods: int) -> float:
    """Factor de fondo de amortización (A/F, i, n) = 1 / (F/A, i, n)."""
    return 1.0 / factor_f_given_a(rate=rate, n_periods=n_periods)


def factor_p_given_a(rate: float, n_periods: int) -> float:
    """Factor de valor presente de serie uniforme (P/A, i, n)."""
    if n_periods <= 0:
        raise ValueError("n_periods debe ser mayor que cero.")
    if math.isclose(rate, 0.0, abs_tol=1e-12):
        return float(n_periods)
    return (((1.0 + rate) ** n_periods) - 1.0) / (rate * ((1.0 + rate) ** n_periods))


def factor_a_given_p(rate: float, n_periods: int) -> float:
    """Factor de recuperación de capital (A/P, i, n) = 1 / (P/A, i, n)."""
    return 1.0 / factor_p_given_a(rate=rate, n_periods=n_periods)


@dataclass
class ControlAlternativeEconomics:
    name: str
    capex_usd: float  # Inversión inicial [USD]
    annual_om_usd: float  # Costo anual de operación y mantenimiento [USD/año]
    annual_energy_usd: float = 0.0  # Costo anual de energía eléctrica/ventilación [USD/año]
    salvage_value_usd: float = 0.0  # Valor de salvamento al final de vida útil [USD]
    life_years: int = 15  # Horizonte / vida útil [años]
    discount_rate_ea: float = 0.12  # Tasa efectiva anual de descuento [decimal]
    inflation_rate_ea: float = 0.0  # Inflación anual opcional [decimal]


def evaluate_alternative(alt: ControlAlternativeEconomics) -> Dict[str, float | str]:
    """Calcula Valor Presente de Costos (VPC) y Costo Anual Equivalente (CAE)."""
    effective_rate = (
        inflation_adjusted_rate(alt.discount_rate_ea, alt.inflation_rate_ea)
        if alt.inflation_rate_ea > 0.0
        else alt.discount_rate_ea
    )
    annual_operating_cost = alt.annual_om_usd + alt.annual_energy_usd

    pv_capex = alt.capex_usd
    pv_opex = annual_operating_cost * factor_p_given_a(effective_rate, alt.life_years)
    pv_salvage = alt.salvage_value_usd * factor_p_given_f(effective_rate, alt.life_years)

    present_cost_usd = pv_capex + pv_opex - pv_salvage
    equivalent_annual_cost_usd = present_cost_usd * factor_a_given_p(
        effective_rate, alt.life_years
    )

    return {
        "alternative": alt.name,
        "effective_rate_pct": effective_rate * 100.0,
        "pv_capex_usd": pv_capex,
        "pv_opex_usd": pv_opex,
        "pv_salvage_usd": pv_salvage,
        "present_cost_usd": present_cost_usd,
        "equivalent_annual_cost_usd": equivalent_annual_cost_usd,
    }


def compare_alternatives(alternatives: List[ControlAlternativeEconomics]) -> List[Dict[str, float | str]]:
    """Evalúa y ordena alternativas de menor a mayor Costo Anual Equivalente (CAE)."""
    results = [evaluate_alternative(alt) for alt in alternatives]
    return sorted(results, key=lambda x: float(x["equivalent_annual_cost_usd"]))


if __name__ == "__main__":
    # Ejemplo ilustrativo usando referencias de costo del Tema 6A y fórmulas del Tema 1
    pes_solo = ControlAlternativeEconomics(
        name="PES Seco (Solo)",
        capex_usd=60000.0 * 0.0471947 * 45000.0 / 100.0,  # ~USD 127,426 ilustrativo
        annual_om_usd=25000.0,
        annual_energy_usd=8000.0,
        salvage_value_usd=10000.0,
        life_years=15,
        discount_rate_ea=0.12,
    )
    ciclon_mas_pes = ControlAlternativeEconomics(
        name="Ciclón + PES Seco (Serie)",
        capex_usd=155000.0,
        annual_om_usd=22000.0,
        annual_energy_usd=9500.0,
        salvage_value_usd=12000.0,
        life_years=15,
        discount_rate_ea=0.12,
    )
    for item in compare_alternatives([pes_solo, ciclon_mas_pes]):
        print(item)