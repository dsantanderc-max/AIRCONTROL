from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.economics import ControlAlternativeEconomics
from core.scenarios import (
    ScenarioConfig,
    compare_scenarios,
    default_scenarios,
    run_scenario,
)
from utils.excel_export import build_excel_report_bytes

st.set_page_config(
    page_title="AIRCONTROL | Control de Material Particulado",
    page_icon="🌬️",
    layout="wide",
)

st.title("AIRCONTROL — Dimensionamiento y Evaluación de Equipos de Control (MP)")
st.caption(
    "Herramienta integrada para propiedades del gas, dimensionamiento de PES y ciclones, "
    "eficiencia fraccional y evaluación técnico-económica."
)

# --- Barra lateral de configuración ---
with st.sidebar:
    st.header("Parámetros de operación")
    preset_mode = st.selectbox(
        "Preset inicial",
        options=["Ejemplo Clase 6A (60,000 CFM)", "Operación Industrial Típica"],
        index=0,
    )

    default_cfm = 60000.0 if "6A" in preset_mode else 45000.0
    default_temp_c = 160.0 if "6A" in preset_mode else 185.0
    default_mu = 2.42e-5 if "6A" in preset_mode else 2.50e-5

    scenario_name = st.text_input("Nombre del escenario activo", value="Escenario Activo")
    flow_cfm = st.number_input("Caudal total Q_T [pie³/min]", min_value=100.0, value=default_cfm, step=1000.0)
    temp_c = st.number_input("Temperatura del gas [°C]", min_value=-20.0, max_value=800.0, value=default_temp_c, step=5.0)
    pressure_atm = st.number_input("Presión absoluta [atm]", min_value=0.2, max_value=5.0, value=1.0, step=0.05)
    moisture_pct = st.number_input("Humedad del gas Bws [%]", min_value=0.0, max_value=80.0, value=5.0, step=0.5)
    viscosity_kg_ms = st.number_input(
        "Viscosidad del gas µ_g [kg/(m·s)]",
        min_value=1e-6,
        max_value=1e-3,
        value=default_mu,
        format="%.2e",
    )
    particle_density = st.number_input(
        "Densidad real de partícula [kg/m³]",
        min_value=100.0,
        max_value=8000.0,
        value=2000.0,
        step=100.0,
    )

    st.divider()
    st.subheader("Emisiones y normatividad")
    inlet_conc = st.number_input("Concentración entrada [mg/m³]", min_value=1.0, value=1500.0, step=50.0)
    standard_limit = st.number_input("Estándar de emisión [mg/m³]", min_value=1.0, value=300.0, step=10.0)
    o2_meas = st.number_input("O₂ medido base seca [%]", min_value=0.0, max_value=20.5, value=8.0, step=0.5)
    o2_ref = st.number_input("O₂ de referencia [%]", min_value=0.0, max_value=20.5, value=11.0, step=0.5)

    st.divider()
    st.subheader("Configuración de equipos")
    use_cyclone = st.checkbox("Incluir Ciclón (precontrol)", value=True)
    use_esp = st.checkbox("Incluir Precipitador Electrostático Seco (PES)", value=True)

    if not use_cyclone and not use_esp:
        st.warning("Activa al menos un equipo de control para evaluar remoción.")

# --- Construcción y ejecución del escenario ---
active_cfg = ScenarioConfig(
    name=scenario_name,
    flow_cfm=flow_cfm,
    temp_c=temp_c,
    pressure_atm=pressure_atm,
    moisture_bws=moisture_pct / 100.0,
    viscosity_kg_ms=viscosity_kg_ms,
    particle_density_kg_m3=particle_density,
    inlet_conc_mg_m3=inlet_conc,
    standard_limit_mg_m3=standard_limit,
    o2_measured_pct=o2_meas,
    o2_ref_pct=o2_ref,
    use_cyclone_precontrol=use_cyclone,
    use_esp=use_esp,
    economics=ControlAlternativeEconomics(
        name=scenario_name,
        capex_usd=730000.0 if (use_cyclone and use_esp) else 650000.0,
        annual_om_usd=23000.0 if (use_cyclone and use_esp) else 25000.0,
        annual_energy_usd=9500.0 if (use_cyclone and use_esp) else 8000.0,
        salvage_value_usd=18000.0 if (use_cyclone and use_esp) else 15000.0,
        life_years=15,
        discount_rate_ea=0.12,
    ),
)

res = run_scenario(active_cfg)
gas = res["gas"]

# --- KPIs superiores ---
k1, k2, k3, k4 = st.columns(4)
k1.metric("Eficiencia Total (%)", f"{float(res['total_efficiency_pct']):.2f}%")
k2.metric("Conc. Salida (mg/m³)", f"{float(res['outlet_conc_mg_m3']):.2f}")
k3.metric("Cumple Estándar", "Sí" if res["complies_standard"] else "No")
k4.metric("UCA (entrada)", f"{float(gas.get('uca') or 0.0):.2f}")

st.divider()

tab1, tab2, tab3 = st.tabs([
    "1. Propiedades del gas y emisión",
    "2. Dimensionamiento PES y Ciclón",
    "3. Comparación de escenarios y Excel",
])

with tab1:
    st.subheader("Propiedades termofísicas y correcciones de corriente gaseosa")
    gas_rows = [{"Parámetro": k, "Valor": v} for k, v in gas.items()]
    st.dataframe(pd.DataFrame(gas_rows), use_container_width=True, hide_index=True)

with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Precipitador Electrostático Seco (PES)")
        if res["esp"] is not None:
            esp_sum = res["esp"]["summary"]
            st.write(f"- **Área requerida ($A_r$):** {esp_sum['required_area_ft2']:.1f} pie²")
            st.write(f"- **Número de placas ($N_p$):** {esp_sum['n_plates']} ({esp_sum['height_ft']:.2f} × {esp_sum['length_ft']:.2f} pies)")
            st.write(f"- **Velocidad del gas ($V_g$):** {esp_sum['gas_velocity_fps']:.2f} pie/s → `{esp_sum['velocity_check']}`")
            st.write(f"- **Voltaje aplicado ($V$):** {esp_sum['voltage_kv']:.2f} kV → `{esp_sum['voltage_check']}`")
            st.write(f"- **Eficiencia PES sola:** {esp_sum['total_efficiency_pct']:.2f}%")
        else:
            st.info("PES desactivado en este escenario.")

    with col_b:
        st.subheader("Ciclón de precontrol")
        if res["cyclone"] is not None:
            cyc_sum = res["cyclone"]["summary"]
            st.write(f"- **Diámetro del cuerpo ($D$):** {cyc_sum['diameter_m']:.2f} m")
            st.write(f"- **Velocidad de entrada ($V_i$):** {cyc_sum['inlet_velocity_m_s']:.2f} m/s")
            st.write(f"- **Diámetro de corte ($d_{50}$):** {cyc_sum['d50_um']:.2f} µm")
            st.write(f"- **Caída de presión ($\Delta P$):** {cyc_sum['delta_p_pa']:.1f} Pa")
            st.write(f"- **Eficiencia Ciclón solo:** {cyc_sum['total_efficiency_pct']:.2f}%")
        else:
            st.info("Ciclón desactivado en este escenario.")

    st.subheader("Eficiencia fraccional por tamaño de partícula")
    comb_df = res["combined_table"]
    st.dataframe(comb_df, use_container_width=True, hide_index=True)

    fig = go.Figure()
    if res["cyclone"] is not None:
        fig.add_trace(
            go.Scatter(
                x=res["cyclone"]["table"]["dp_um"],
                y=res["cyclone"]["table"]["eta_pct"],
                mode="lines+markers",
                name="Ciclón (%)",
            )
        )
    if res["esp"] is not None:
        fig.add_trace(
            go.Scatter(
                x=res["esp"]["table"]["dp_um"],
                y=res["esp"]["table"]["eta_pct"],
                mode="lines+markers",
                name="PES (%)",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=comb_df["dp_um"],
            y=comb_df["eta_comb_pct"],
            mode="lines+markers",
            name="Combinada (%)",
        )
    )
    fig.update_layout(
        xaxis_title="Diámetro de partícula dp (µm)",
        yaxis_title="Eficiencia fraccional (%)",
        yaxis_range=[0, 105],
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Comparación de escenarios base y exportación")
    base_list = default_scenarios()
    comp_df = compare_scenarios(base_list)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    excel_bytes = build_excel_report_bytes(
        comparison_df=comp_df,
        scenario_result=res,
    )
    st.download_button(
        label="Descargar reporte completo (.xlsx)",
        data=excel_bytes,
        file_name="aircontrol_evaluacion.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )