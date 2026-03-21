from importing import export_roots_to_csv
import numpy as np
import pandas as pd

#####################
# Ajuste 

def _close(a, b, tol=1e-9):
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(float(a) - float(b)) < tol


def _clean_gas_name(g):
    if pd.isna(g):
        return None
    g = str(g).strip()
    return g if g else None


def _normalize_fraction(c):
    """
    Normaliza composiciones:
    - 1.0 -> 100.0
    - 100.0 -> 100.0
    - 0.7 se queda 0.7 (no se toca)
    """
    if pd.isna(c):
        return None
    c = float(c)
    if _close(c, 1.0):
        return 100.0
    return c


def _is_pure_mix(comp1, comp2):
    c1 = _normalize_fraction(comp1)
    c2_raw = comp2
    c2 = 0.0 if pd.isna(c2_raw) else float(c2_raw)

    return (_close(c1, 100.0) or _close(c1, 1.0)) and (_close(c2, 0.0) or pd.isna(c2_raw))


def _canonical_mix(gas1, gas2, comp1, comp2):
    """
    Devuelve una representación canónica de la mezcla:
    - gas puro: (("Ar", 100.0),)
    - mezcla binaria: (("Ar", 70.0), ("CO2", 30.0))
    Ordenada por nombre de gas para que Ar/CO2 = CO2/Ar.
    """
    g1 = _clean_gas_name(gas1)
    g2 = _clean_gas_name(gas2)
    c1 = _normalize_fraction(comp1)
    c2 = None if pd.isna(comp2) else float(comp2)

    if _is_pure_mix(c1, c2):
        return ((g1, 100.0),)

    comps = []
    if g1 is not None and c1 is not None and not _close(c1, 0.0):
        comps.append((g1, float(c1)))
    if g2 is not None and c2 is not None and not _close(c2, 0.0):
        # si viniera como 1.0 por algún motivo, aquí no lo fuerzo a 100 salvo que sea gas puro
        comps.append((g2, float(c2)))

    comps = sorted(comps, key=lambda x: x[0])
    return tuple(comps)


def _row_matches_mix(row, target_mix, comp_tol=1e-6):
    row_mix = _canonical_mix(
        row.get("gas1"),
        row.get("gas2"),
        row.get("composition1"),
        row.get("composition2"),
    )

    if len(row_mix) != len(target_mix):
        return False

    for (rg, rc), (tg, tc) in zip(row_mix, target_mix):
        if rg != tg:
            return False
        if abs(float(rc) - float(tc)) > comp_tol:
            return False

    return True


def select_mix(df, gas1, composition1, gas2=None, composition2=0.0, comp_tol=1e-6):
    """
    Filtra el DataFrame para una mezcla dada.
    Soporta:
    - gas puro: gas1="Ar", composition1=100
    - mezcla: gas1="Ar", composition1=70, gas2="CO2", composition2=30
    Ignora el orden gas1/gas2 en el DataFrame.
    """
    target_mix = _canonical_mix(gas1, gas2, composition1, composition2)
    mask = df.apply(lambda row: _row_matches_mix(row, target_mix, comp_tol=comp_tol), axis=1)
    return df[mask].copy()


def fit_townsend_AB(df_mix, e_col="electricField", p_col="pressure", alpha_col="alpha"):
    """
    Ajusta la ley:
        alpha/p = A * exp(-B * p/E)

    Devuelve un dict con:
        A, B, slope, intercept, r2, n_points
    """
    data = df_mix[[e_col, p_col, alpha_col]].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    data = data[(data[e_col] > 0) & (data[p_col] > 0) & (data[alpha_col] > 0)].copy()

    if len(data) < 2:
        raise ValueError("No hay suficientes puntos válidos para ajustar A y B.")

    x = data[p_col].to_numpy(dtype=float) / data[e_col].to_numpy(dtype=float)          # p/E
    y = np.log(data[alpha_col].to_numpy(dtype=float) / data[p_col].to_numpy(dtype=float))  # ln(alpha/p)

    slope, intercept = np.polyfit(x, y, 1)

    # y = intercept + slope * x = ln(A) - B * x
    A = np.exp(intercept)
    B = -slope

    y_fit = intercept + slope * x
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "A": A,
        "B": B,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "n_points": len(data),
    }


def predict_alpha(E, p, A, B):
    """
    Predice alpha usando:
        alpha = A * p * exp(-B * p / E)
    """
    E = np.asarray(E, dtype=float)
    p = np.asarray(p, dtype=float)
    return A * p * np.exp(-B * (p / E))


def predict_alpha_from_fit(E, p, fit_result):
    return predict_alpha(E, p, fit_result["A"], fit_result["B"])


def predict_E(alpha, p, A, B):
    """
    Inversa analítica para E, con alpha y p dados:
        E = B p / ln(A p / alpha)

    Requiere alpha > 0 y A p > alpha.
    """
    alpha = np.asarray(alpha, dtype=float)
    p = np.asarray(p, dtype=float)

    arg = (A * p) / alpha
    if np.any(arg <= 1.0):
        raise ValueError("Hace falta A*p > alpha para que la inversa en E sea real.")
    return B * p / np.log(arg)

#####################
# main 
"""
folder = "/ruta/a/tu/carpeta_con_roots"
output_csv = "gas_data.csv"
tree_name = "dataOfGas"
recursive = True

df = export_roots_to_csv(
    folder=folder,
    output_csv=output_csv,
    tree_name=tree_name,
    recursive=recursive
)
"""