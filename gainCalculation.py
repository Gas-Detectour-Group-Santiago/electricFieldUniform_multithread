import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares, brentq

from importing import export_roots_to_csv


TORR_PER_BAR = 750.061683
BAR_PER_TORR = 1.0 / TORR_PER_BAR


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
    - 1.0   -> 100.0
    - 100.0 -> 100.0
    - otros valores se dejan tal cual
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


def _prepare_townsend_data(df_mix, e_col="electricField", p_col="pressure", alpha_col="alpha"):
    data = df_mix[[e_col, p_col, alpha_col]].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data[(data[e_col] > 0) & (data[p_col] > 0) & (data[alpha_col] > 0)].copy()

    if len(data) < 4:
        raise ValueError("No hay suficientes puntos válidos para el ajuste.")

    E = data[e_col].to_numpy(dtype=float)
    p = data[p_col].to_numpy(dtype=float)
    alpha = data[alpha_col].to_numpy(dtype=float)

    X = E / p
    ylog = np.log(alpha / p)

    return data, E, p, alpha, X, ylog


def fit_townsend_AB(df_mix, e_col="electricField", p_col="pressure", alpha_col="alpha"):
    """
    Ajuste clásico de Korff/Townsend:
        alpha / p = A * exp(-B * p / E)

    Equivalente a:
        ln(alpha/p) = ln(A) - B * (p/E)
    """
    data = df_mix[[e_col, p_col, alpha_col]].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data[(data[e_col] > 0) & (data[p_col] > 0) & (data[alpha_col] > 0)].copy()

    if len(data) < 2:
        raise ValueError("No hay suficientes puntos válidos para ajustar A y B.")

    x = data[p_col].to_numpy(dtype=float) / data[e_col].to_numpy(dtype=float)
    y = np.log(data[alpha_col].to_numpy(dtype=float) / data[p_col].to_numpy(dtype=float))

    slope, intercept = np.polyfit(x, y, 1)

    A = np.exp(intercept)
    B = -slope

    y_fit = intercept + slope * x
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    alpha_fit = predict_alpha_korff(
        data[e_col].to_numpy(dtype=float),
        data[p_col].to_numpy(dtype=float),
        A,
        B,
    )
    alpha_true = data[alpha_col].to_numpy(dtype=float)
    rel_err = (alpha_fit - alpha_true) / alpha_true
    rmse_rel = np.sqrt(np.mean(rel_err ** 2))
    mape = np.mean(np.abs(rel_err))

    return {
        "model": "korff",
        "A": float(A),
        "B": float(B),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2_log": float(r2),
        "rmse_rel": float(rmse_rel),
        "mape": float(mape),
        "n_points": len(data),
    }


def _predict_alpha_generalized_raw(E, p, A, B, m, n):
    """
    Modelo generalizado:
        alpha / p = A * (E/p)^m * exp(-(B / (E/p))^n)
    """
    E = np.asarray(E, dtype=float)
    p = np.asarray(p, dtype=float)
    X = E / p
    return p * A * np.power(X, m) * np.exp(-np.power(B / X, n))


def fit_townsend_generalized(
    df_mix,
    e_col="electricField",
    p_col="pressure",
    alpha_col="alpha",
    loss="soft_l1",
    f_scale=0.15,
    max_nfev=50000,
):
    """
    Ajusta el modelo generalizado:

        alpha / p = A * (E/p)^m * exp(-(B / (E/p))^n)

    equivalente a

        ln(alpha/p) = ln(A) + m ln(E/p) - (B / (E/p))^n

    El ajuste se hace en log-space y con pérdida robusta.
    """
    data, E, p, alpha, X, ylog = _prepare_townsend_data(
        df_mix, e_col=e_col, p_col=p_col, alpha_col=alpha_col
    )

    try:
        seed = fit_townsend_AB(df_mix, e_col=e_col, p_col=p_col, alpha_col=alpha_col)
        logA0 = np.log(max(seed["A"], 1e-30))
        logB0 = np.log(max(seed["B"], 1e-30))
        m0 = 0.5
        logn0 = np.log(1.0)
    except Exception:
        logA0 = 0.0
        logB0 = np.log(max(np.median(X), 1e-12))
        m0 = 0.5
        logn0 = np.log(1.0)

    p0 = np.array([logA0, logB0, m0, logn0], dtype=float)

    # theta = [logA, logB, m, logn]
    lower = np.array([-100.0, -100.0, -10.0, np.log(0.2)], dtype=float)
    upper = np.array([+100.0, +100.0, +10.0, np.log(6.0)], dtype=float)

    def residuals(theta):
        logA, logB, m, logn = theta
        B = np.exp(logB)
        n = np.exp(logn)
        yhat = logA + m * np.log(X) - np.power(B / X, n)
        return yhat - ylog

    res = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        x_scale="jac",
    )

    logA, logB, m, logn = res.x
    A = float(np.exp(logA))
    B = float(np.exp(logB))
    m = float(m)
    n = float(np.exp(logn))

    alpha_fit = _predict_alpha_generalized_raw(E, p, A, B, m, n)
    ylog_fit = np.log(alpha_fit / p)

    ss_res_log = np.sum((ylog - ylog_fit) ** 2)
    ss_tot_log = np.sum((ylog - np.mean(ylog)) ** 2)
    r2_log = 1.0 - ss_res_log / ss_tot_log if ss_tot_log > 0 else np.nan

    rel_err = (alpha_fit - alpha) / alpha
    rmse_rel = np.sqrt(np.mean(rel_err ** 2))
    mape = np.mean(np.abs(rel_err))

    return {
        "model": "generalized",
        "A": A,
        "B": B,
        "m": m,
        "n": n,
        "r2_log": float(r2_log),
        "rmse_rel": float(rmse_rel),
        "mape": float(mape),
        "cost": float(res.cost),
        "success": bool(res.success),
        "message": res.message,
        "n_points": len(data),
    }


def predict_alpha_korff(E, p, A, B):
    """
    Modelo clásico:
        alpha = A * p * exp(-B * p / E)
    """
    E = np.asarray(E, dtype=float)
    p = np.asarray(p, dtype=float)
    return A * p * np.exp(-B * (p / E))


def predict_alpha_generalized(E, p, A, B, m, n):
    return _predict_alpha_generalized_raw(E, p, A, B, m, n)


def predict_alpha_from_fit(E, p, fit_result):
    model = fit_result.get("model", "generalized")

    if model == "generalized":
        return predict_alpha_generalized(
            E,
            p,
            fit_result["A"],
            fit_result["B"],
            fit_result["m"],
            fit_result["n"],
        )

    if model == "korff":
        return predict_alpha_korff(
            E,
            p,
            fit_result["A"],
            fit_result["B"],
        )

    raise ValueError(f"Modelo de ajuste desconocido: {model}")


def alpha_to_gain(alpha, gap_mm):
    """
    G = exp(alpha * d), con d en cm.
    """
    alpha = np.asarray(alpha, dtype=float)
    gap_cm = np.asarray(gap_mm, dtype=float) * 0.1
    return np.exp(alpha * gap_cm)


def gain_to_alpha(gain, gap_mm):
    """
    alpha = ln(G) / d, con d en cm.
    """
    gain = np.asarray(gain, dtype=float)
    gap_cm = np.asarray(gap_mm, dtype=float) * 0.1

    if np.any(gain <= 0):
        raise ValueError("La ganancia debe ser > 0.")
    if np.any(gap_cm <= 0):
        raise ValueError("El gap debe ser > 0.")

    return np.log(gain) / gap_cm


def predict_gain(E, p, gap_mm, fit_result):
    alpha = predict_alpha_from_fit(E, p, fit_result)
    return alpha_to_gain(alpha, gap_mm)


def required_gap_for_gain(gain, E, p, fit_result):
    """
    d = ln(G) / alpha -> devuelve gap en mm
    """
    alpha = predict_alpha_from_fit(E, p, fit_result)

    if np.any(alpha <= 0):
        raise ValueError("Alpha <= 0: no se puede calcular un gap físico.")

    gap_cm = np.log(np.asarray(gain, dtype=float)) / alpha
    return gap_cm * 10.0


def required_E_for_gain(gain, p, gap_mm, fit_result, e_min=1.0, e_max=1e6):
    """
    Calcula numéricamente el E necesario para lograr una ganancia dada,
    resolviendo alpha(E, p) = ln(G)/d.
    Devuelve E en las mismas unidades que uses en electricField.
    """
    alpha_target = gain_to_alpha(gain, gap_mm)
    alpha_target = np.asarray(alpha_target, dtype=float)
    p = np.asarray(p, dtype=float)

    scalar_output = False
    if alpha_target.ndim == 0:
        alpha_target = np.array([alpha_target], dtype=float)
        scalar_output = True

    if p.ndim == 0:
        p = np.full(alpha_target.shape, float(p))
    elif p.shape != alpha_target.shape:
        raise ValueError("p y gain/gap deben ser compatibles en forma.")

    out = []

    for alpha_t, p_t in zip(alpha_target, p):
        if alpha_t <= 0:
            raise ValueError("Alpha objetivo <= 0.")

        def f(E):
            return predict_alpha_from_fit(E, p_t, fit_result) - alpha_t

        lo = float(e_min)
        hi = float(e_max)

        flo = f(lo)
        fhi = f(hi)

        n_expand = 0
        while fhi < 0 and n_expand < 60:
            hi *= 2.0
            fhi = f(hi)
            n_expand += 1

        if flo > 0:
            raise ValueError("El e_min ya da una alpha mayor que la objetivo.")
        if fhi < 0:
            raise ValueError("No se ha podido encerrar la solución aumentando e_max.")

        out.append(brentq(f, lo, hi))

    out = np.asarray(out)
    return out[0] if scalar_output else out


def mix_label(gas1, composition1, gas2=None, composition2=0.0):
    mix = _canonical_mix(gas1, gas2, composition1, composition2)

    if len(mix) == 1:
        return f"{mix[0][0]} {mix[0][1]:.0f}%"

    parts = [f"{gas} {comp:.0f}%" for gas, comp in mix]
    return " / ".join(parts)


def mix_slug(gas1, composition1, gas2=None, composition2=0.0):
    label = mix_label(gas1, composition1, gas2, composition2)
    label = label.replace("%", "pct")
    label = re.sub(r"\s*/\s*", "_", label)
    label = re.sub(r"\s+", "_", label)
    label = re.sub(r"[^A-Za-z0-9_.-]", "", label)
    return label


def plot_alpha_fit_by_pressure(
    df_mix,
    fit_result,
    gas1,
    composition1,
    gas2=None,
    composition2=0.0,
    pdf_dir="fitPlots",
    pdf_path=None,
    top_n_pressures=3,
):
    """
    Guarda un PDF con alpha vs E para las top-N presiones con más datos.
    Se muestran los puntos y la curva del ajuste global.
    """
    data = df_mix[["electricField", "pressure", "alpha"]].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data[(data["electricField"] > 0) & (data["pressure"] > 0) & (data["alpha"] > 0)].copy()

    if data.empty:
        raise ValueError("No hay datos válidos para graficar alpha vs E.")

    pressure_counts = data["pressure"].value_counts()
    top_pressures = pressure_counts.index[:top_n_pressures].tolist()

    if pdf_path is None:
        os.makedirs(pdf_dir, exist_ok=True)
        pdf_name = f"alpha_fit_{mix_slug(gas1, composition1, gas2, composition2)}.pdf"
        pdf_path = os.path.join(pdf_dir, pdf_name)
    else:
        pdf_dirname = os.path.dirname(os.path.abspath(pdf_path))
        if pdf_dirname:
            os.makedirs(pdf_dirname, exist_ok=True)

    plt.figure(figsize=(8.5, 6.0))

    for pressure_value in top_pressures:
        subset = data[np.isclose(data["pressure"], pressure_value)]
        subset = subset.sort_values("electricField")

        plt.scatter(
            subset["electricField"].to_numpy(),
            subset["alpha"].to_numpy(),
            label=f"Datos p = {pressure_value * BAR_PER_TORR:.3f} Bar (N={len(subset)})",
        )

        e_min = subset["electricField"].min()
        e_max = subset["electricField"].max()
        e_grid = np.linspace(e_min, e_max, 500)
        alpha_fit = predict_alpha_from_fit(e_grid, pressure_value, fit_result)

        plt.plot(
            e_grid,
            alpha_fit,
            label=f"Ajuste p = {pressure_value * BAR_PER_TORR:.3f} Bar",
        )

    plt.xlabel("Electric field [V/cm]")
    plt.ylabel(r"Townsend coefficient $\alpha$ [1/cm]")

    label = mix_label(gas1, composition1, gas2, composition2)

    if fit_result.get("model") == "generalized":
        title = (
            f"{label}\n"
            f"A={fit_result['A']:.4g}, B={fit_result['B']:.4g}, "
            f"m={fit_result['m']:.4g}, n={fit_result['n']:.4g}, "
            f"$R^2_{{\\log}}$={fit_result['r2_log']:.4f}, "
            f"MAPE={100 * fit_result['mape']:.1f}%, "
            f"N={fit_result['n_points']}"
        )
    else:
        title = (
            f"{label}\n"
            f"A={fit_result['A']:.4g}, B={fit_result['B']:.4g}, "
            f"$R^2_{{\\log}}$={fit_result['r2_log']:.4f}, "
            f"MAPE={100 * fit_result['mape']:.1f}%, "
            f"N={fit_result['n_points']}"
        )

    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()

    return pdf_path


def fit_mix_from_dataframe(
    df,
    gas1,
    composition1,
    gas2=None,
    composition2=0.0,
    model="generalized",
    make_pdf=False,
    pdf_dir="fitPlots",
    pdf_path=None,
):
    df_mix = select_mix(df, gas1, composition1, gas2, composition2)

    if df_mix.empty:
        raise ValueError(
            f"No hay datos en el CSV para la mezcla {mix_label(gas1, composition1, gas2, composition2)}."
        )

    if model == "generalized":
        fit_result = fit_townsend_generalized(df_mix)
    elif model == "korff":
        fit_result = fit_townsend_AB(df_mix)
    else:
        raise ValueError("model debe ser 'generalized' o 'korff'.")

    result = {
        "fit": fit_result,
        "df_mix": df_mix,
        "mix_label": mix_label(gas1, composition1, gas2, composition2),
        "pdf_path": None,
    }

    if make_pdf:
        result["pdf_path"] = plot_alpha_fit_by_pressure(
            df_mix=df_mix,
            fit_result=fit_result,
            gas1=gas1,
            composition1=composition1,
            gas2=gas2,
            composition2=composition2,
            pdf_dir=pdf_dir,
            pdf_path=pdf_path,
        )

    return result


def fit_mix_from_csv(
    csv_path,
    gas1,
    composition1,
    gas2=None,
    composition2=0.0,
    model="generalized",
    make_pdf=False,
    pdf_dir="fitPlots",
    pdf_path=None,
):
    df = pd.read_csv(csv_path)
    return fit_mix_from_dataframe(
        df=df,
        gas1=gas1,
        composition1=composition1,
        gas2=gas2,
        composition2=composition2,
        model=model,
        make_pdf=make_pdf,
        pdf_dir=pdf_dir,
        pdf_path=pdf_path,
    )


def update_csv_and_fit_mix(
    root_folder,
    csv_path,
    gas1,
    composition1,
    gas2=None,
    composition2=0.0,
    tree_name="dataOfGas",
    recursive=True,
    model="generalized",
    make_pdf=False,
    pdf_dir="fitPlots",
    pdf_path=None,
):
    """
    Flujo completo:
    1) Actualiza/crea el CSV desde los ROOT
    2) Selecciona la mezcla
    3) Ajusta el modelo elegido
    4) Opcionalmente guarda el PDF del ajuste
    """
    df = export_roots_to_csv(
        folder=root_folder,
        output_csv=csv_path,
        tree_name=tree_name,
        recursive=recursive,
    )

    return fit_mix_from_dataframe(
        df=df,
        gas1=gas1,
        composition1=composition1,
        gas2=gas2,
        composition2=composition2,
        model=model,
        make_pdf=make_pdf,
        pdf_dir=pdf_dir,
        pdf_path=pdf_path,
    )


def pressure_bar_to_torr(pressure_bar):
    return float(pressure_bar) * TORR_PER_BAR