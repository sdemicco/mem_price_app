import pandas as pd
import numpy as np
import pickle


def load_model(model_path):

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


def run_scenario(
    df_hist,
    model,
    shock_gas=0,
    shock_demanda=0,
    shock_hidro=0,
    shock_renov=0,
    base_method="average"
):

    df = df_hist.copy()

    df["MES"] = df["indice_tiempo"].dt.month
    df["YEAR"] = df["indice_tiempo"].dt.year


    # -------------------------
    # ESTACIONALIDAD
    # -------------------------

    demanda_season = df.groupby("MES")["DemandaTotal"].mean()
    demanda_season = demanda_season / demanda_season.mean()

    hidro_season = df.groupby("MES")["Renovable HIDRO > 50"].mean()
    hidro_season = hidro_season / hidro_season.mean()

    renov_season = df.groupby("MES")["Generacion Renovable"].mean()
    renov_season = renov_season / renov_season.mean()


    # -------------------------
    # BASELINE
    # -------------------------

    if base_method == "average":

        base = df.groupby("MES").mean(numeric_only=True).reset_index()

    elif base_method == "last_year":

        last_year = df["YEAR"].max() - 1

        base = (
            df[df["YEAR"] == last_year]
            .sort_values("MES")
            .reset_index(drop=True)
        )


    # -------------------------
    # ESCENARIO
    # -------------------------

    base["gas_scenario"] = (
        base["precio GAS NATURAL"] * (1 + shock_gas)
    )

    base["demanda_scenario"] = (
        base["DemandaTotal"] *
        (1 + shock_demanda * base["MES"].map(demanda_season))
    )

    base["hidro_scenario"] = (
        base["Renovable HIDRO > 50"] *
        (1 + shock_hidro * base["MES"].map(hidro_season))
    )

    base["renov_scenario"] = (
        base["Generacion Renovable"] *
        (1 + shock_renov * base["MES"].map(renov_season))
    )


    # -------------------------
    # VARIABLES DEL MODELO
    # -------------------------

    base["log_Gas"] = np.log(base["gas_scenario"])

    base["demanda_rel"] = base["demanda_scenario"] / base["DemandaTotal"]
    base["log_demanda_rel"] = np.log(base["demanda_rel"])

    base["hidro_rel"] = base["hidro_scenario"] / base["Renovable HIDRO > 50"]
    base["log_hidro_rel"] = np.log(base["hidro_rel"])

    base["renov_share"] = (
        base["renov_scenario"] /
        (base["renov_scenario"] + base["Generacion Termica"])
    )

    base["log_renov_share"] = np.log(base["renov_share"])


    # -------------------------
    # PREDICCION
    # -------------------------

    base["log_pred"] = model.predict(base)

    base["precio_pred"] = np.exp(base["log_pred"])


    # -------------------------
    # INCERTIDUMBRE MODELO
    # -------------------------

    sigma = model.resid.std()

    uncertainty_pct = (np.exp(sigma) - 1) * 100


    # -------------------------
    # PRECIO ANUAL
    # -------------------------

    precio_anual = base["precio_pred"].mean()

    return base[["MES", "precio_pred"]], precio_anual, uncertainty_pct