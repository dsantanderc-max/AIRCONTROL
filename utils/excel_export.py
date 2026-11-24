from __future__ import annotations

from io import BytesIO
from typing import Dict, List, Optional

import pandas as pd


def build_excel_report_bytes(
    comparison_df: pd.DataFrame,
    scenario_result: Dict[str, object],
    extra_tables: Optional[Dict[str, pd.DataFrame]] = None,
) -> bytes:
    """Genera un archivo Excel (.xlsx) en memoria con los resultados de AIRCONTROL."""
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        comparison_df.to_excel(writer, sheet_name="Comparacion_Escenarios", index=False)

        gas_dict = scenario_result.get("gas")
        if isinstance(gas_dict, dict):
            gas_df = pd.DataFrame(
                [{"Parametro": k, "Valor": v} for k, v in gas_dict.items()]
            )
            gas_df.to_excel(writer, sheet_name="Propiedades_Gas", index=False)

        esp_out = scenario_result.get("esp")
        if isinstance(esp_out, dict):
            esp_sum = esp_out.get("summary")
            if isinstance(esp_sum, dict):
                pd.DataFrame(
                    [{"Parametro": k, "Valor": v} for k, v in esp_sum.items()]
                ).to_excel(writer, sheet_name="Resumen_PES", index=False)

            esp_tbl = esp_out.get("table")
            if isinstance(esp_tbl, pd.DataFrame):
                esp_tbl.to_excel(writer, sheet_name="Fraccional_PES", index=False)

        cyc_out = scenario_result.get("cyclone")
        if isinstance(cyc_out, dict):
            cyc_sum = cyc_out.get("summary")
            if isinstance(cyc_sum, dict):
                pd.DataFrame(
                    [{"Parametro": k, "Valor": v} for k, v in cyc_sum.items()]
                ).to_excel(writer, sheet_name="Resumen_Ciclon", index=False)

            cyc_tbl = cyc_out.get("table")
            if isinstance(cyc_tbl, pd.DataFrame):
                cyc_tbl.to_excel(writer, sheet_name="Fraccional_Ciclon", index=False)

        comb_tbl = scenario_result.get("combined_table")
        if isinstance(comb_tbl, pd.DataFrame):
            comb_tbl.to_excel(writer, sheet_name="Eficiencia_Combinada", index=False)

        if extra_tables:
            for sheet_name, df in extra_tables.items():
                safe_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=safe_name, index=False)

    output.seek(0)
    return output.getvalue()


def save_excel_report(
    filepath: str,
    comparison_df: pd.DataFrame,
    scenario_result: Dict[str, object],
    extra_tables: Optional[Dict[str, pd.DataFrame]] = None,
) -> str:
    """Guarda el reporte Excel (.xlsx) en disco y devuelve la ruta."""
    content = build_excel_report_bytes(
        comparison_df=comparison_df,
        scenario_result=scenario_result,
        extra_tables=extra_tables,
    )
    with open(filepath, "wb") as f:
        f.write(content)
    return filepath


if __name__ == "__main__":
    from core.scenarios import compare_scenarios, default_scenarios, run_scenario

    scenarios = default_scenarios()
    comp_df = compare_scenarios(scenarios)
    res_0 = run_scenario(scenarios[0])
    out_path = save_excel_report("aircontrol_reporte_prueba.xlsx", comp_df, res_0)
    print(f"Reporte generado exitosamente en: {out_path}")