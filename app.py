import streamlit as st
import pandas as pd

from model.scenario_engine import load_model, run_scenario


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="MEM Price Scenario Simulator",
    layout="wide"
)


st.title("MEM Price Scenario Simulator")

st.markdown(
"""
This tool simulates how structural drivers affect the **Monómico Total (Local)**  
price of the Argentine electricity market.

Adjust the assumptions on the left to evaluate different market scenarios.
"""
)

st.divider()


# -------------------------------------------------
# RESET FUNCTION
# -------------------------------------------------

def reset_scenario():

    st.session_state.gas_change = 0
    st.session_state.demand_change = 0
    st.session_state.hydro_change = 0
    st.session_state.renew_change = 0


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("data/variables_relevantes_MEM.csv")

    df["indice_tiempo"] = pd.to_datetime(df["indice_tiempo"])

    df["DemandaIndustrialyComercial"] = (
        df["Demanda Comercial"] +
        df["Gran Demanda Industrial/Comercial"]
    )

    df["DemandaTotal"] = (
        df["DemandaIndustrialyComercial"] +
        df["Demanda Residencial"]
    )

    return df


df_hist = load_data()


# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

model = load_model("model/trained_model.pkl")


# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------

if "gas_change" not in st.session_state:
    st.session_state.gas_change = 0

if "demand_change" not in st.session_state:
    st.session_state.demand_change = 0

if "hydro_change" not in st.session_state:
    st.session_state.hydro_change = 0

if "renew_change" not in st.session_state:
    st.session_state.renew_change = 0


# -------------------------------------------------
# SIDEBAR INPUTS
# -------------------------------------------------

st.sidebar.header("Scenario Assumptions")

st.sidebar.markdown(
"Introduce annual variation assumptions for each driver."
)

gas_input = st.sidebar.slider(
    "Gas price change",
    -50,
    100,
    key="gas_change",
    format="%d%%",
    help="Annual change applied to gas prices"
)

demanda_input = st.sidebar.slider(
    "Demand change",
    -20,
    20,
    key="demand_change",
    format="%d%%",
    help="Annual change in electricity demand"
)

hidro_input = st.sidebar.slider(
    "Hydro generation change",
    -50,
    50,
    key="hydro_change",
    format="%d%%",
)

renov_input = st.sidebar.slider(
    "Renewables generation change",
    -50,
    50,
    key="renew_change",
    format="%d%%",
)

st.sidebar.button(
    "Reset scenario",
    on_click=reset_scenario
)

base_method = st.sidebar.radio(
    "Baseline reference",
    ["Historical average", "Last year"]
)

method = "average"

if base_method == "Last year":
    method = "last_year"


# -------------------------------------------------
# RUN SCENARIOS
# -------------------------------------------------

precio_base, precio_anual_base, uncertainty_pct = run_scenario(
    df_hist,
    model,
    0,0,0,0,
    method
)

precio_escenario, precio_anual_esc, _ = run_scenario(
    df_hist,
    model,
    gas_input/100,
    demanda_input/100,
    hidro_input/100,
    renov_input/100,
    method
)


# -------------------------------------------------
# ANNUAL PRICE RESULTS
# -------------------------------------------------

st.header("Annual Price Result")

delta_usd = precio_anual_esc - precio_anual_base
delta_pct = (precio_anual_esc/precio_anual_base - 1) * 100

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Base Annual Price",
        f"{precio_anual_base:.2f} USD/MWh"
    )

    st.caption(
        f"Model uncertainty: ±{uncertainty_pct:.1f}%"
    )


with col2:

    st.metric(
        "Scenario Annual Price",
        f"{precio_anual_esc:.2f} USD/MWh",
        delta=f"{delta_pct:.1f}%"
    )

    st.caption(
        f"Change vs base: {delta_usd:.2f} USD/MWh"
    )


st.divider()


# -------------------------------------------------
# MONTHLY PRICE CURVE
# -------------------------------------------------

st.header("Monthly Price Curve")

st.caption(
"The chart shows the expected monthly price under the base case and the selected scenario."
)

mes_labels = [
"Jan","Feb","Mar","Apr","May","Jun",
"Jul","Aug","Sep","Oct","Nov","Dec"
]

chart_df = pd.DataFrame()

chart_df["MES"] = precio_base["MES"]

chart_df["Mes"] = chart_df["MES"].apply(
    lambda x: mes_labels[int(x)-1]
)

chart_df["Base"] = precio_base["precio_pred"]
chart_df["Scenario"] = precio_escenario["precio_pred"]

chart_df = chart_df.set_index("Mes")

st.line_chart(chart_df[["Base","Scenario"]])


st.divider()


# -------------------------------------------------
# DRIVER DECOMPOSITION
# -------------------------------------------------

st.header("Drivers of Price Change")

drivers = {}

precio_no_gas, p_no_gas,_ = run_scenario(
    df_hist, model,
    0,
    demanda_input/100,
    hidro_input/100,
    renov_input/100,
    method
)

drivers["Gas"] = precio_anual_esc - p_no_gas


precio_no_dem, p_no_dem,_ = run_scenario(
    df_hist, model,
    gas_input/100,
    0,
    hidro_input/100,
    renov_input/100,
    method
)

drivers["Demand"] = precio_anual_esc - p_no_dem


precio_no_hidro, p_no_hidro,_ = run_scenario(
    df_hist, model,
    gas_input/100,
    demanda_input/100,
    0,
    renov_input/100,
    method
)

drivers["Hydro"] = precio_anual_esc - p_no_hidro


precio_no_renov, p_no_renov,_ = run_scenario(
    df_hist, model,
    gas_input/100,
    demanda_input/100,
    hidro_input/100,
    0,
    method
)

drivers["Renewables"] = precio_anual_esc - p_no_renov


drivers_df = pd.DataFrame({

    "Variable": list(drivers.keys()),
    "Impact USD": list(drivers.values())

})


st.bar_chart(drivers_df.set_index("Variable"))

st.dataframe(drivers_df)


st.divider()


# -------------------------------------------------
# MONTHLY TABLE
# -------------------------------------------------

st.header("Monthly Prices")

st.dataframe(chart_df)