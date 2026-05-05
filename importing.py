
import os
import glob
import numpy as np
import pandas as pd
import uproot

TORR_PER_BAR = 750.061683
DEFAULT_MIN_NPE_FOR_ALPHA = 100

CSV_COLUMNS = [
    "file", "sourceFiles", "nRuns",
    "gas1", "gas2",
    "composition1", "composition2",
    "electricField", "gap", "pressure", "pressureBar", "temperature",
    "npe", "neTotal", "niTotal", "neMean", "niMean",
    "gainSim", "alphaEff", "alphaFromNe", "alphaFromNi",
    "vz", "validForAlpha", "alphaSource",
]


def _to_python_scalar(x):
    """Convierte valores de uproot/numpy/bytes a escalares Python limpios."""
    if x is None:
        return None

    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="ignore")

    if hasattr(x, "item") and not isinstance(x, str):
        try:
            return x.item()
        except Exception:
            pass

    return x


def _value_at(arr, i, default=np.nan):
    if arr is None:
        return default
    try:
        if len(arr) == 0:
            return default
        j = i if i < len(arr) else 0
        return _to_python_scalar(arr[j])
    except Exception:
        return default


def _read_first_existing_branch(tree, candidates):
    """
    Devuelve el array de la primera branch existente entre candidates.
    Si no existe ninguna, devuelve None.
    """
    tree_keys = set(tree.keys())

    for name in candidates:
        if name in tree_keys:
            try:
                return tree[name].array(library="np")
            except Exception:
                try:
                    arr = tree[name].array(library="ak")
                    return np.array(arr.to_list(), dtype=object)
                except Exception:
                    return None
    return None


def _safe_float(x, default=np.nan):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=0):
    try:
        if x is None or pd.isna(x):
            return default
        return int(round(float(x)))
    except Exception:
        return default


def _fallback_avalanche_stats(root_file):
    """
    Recupera npe, ne y ni desde dataPerPrimaryElectron cuando el ROOT todavía
    no tiene esos campos resumidos en dataOfGas.
    """
    stats = {
        "npe": np.nan,
        "neTotal": np.nan,
        "niTotal": np.nan,
        "neMean": np.nan,
        "niMean": np.nan,
        "gainSim": np.nan,
    }

    if "dataPerPrimaryElectron" not in root_file:
        return stats

    try:
        tree = root_file["dataPerPrimaryElectron"]
        ne_arr = _read_first_existing_branch(tree, ["nElectrons", "ne", "nelec"])
        ni_arr = _read_first_existing_branch(tree, ["nIons", "ni", "nion"])

        n_candidates = []
        if ne_arr is not None:
            n_candidates.append(len(ne_arr))
        if ni_arr is not None:
            n_candidates.append(len(ni_arr))
        if not n_candidates:
            return stats

        npe = int(max(n_candidates))
        ne_total = float(np.nansum(ne_arr.astype(float))) if ne_arr is not None else np.nan
        ni_total = float(np.nansum(ni_arr.astype(float))) if ni_arr is not None else np.nan

        stats["npe"] = npe
        stats["neTotal"] = ne_total
        stats["niTotal"] = ni_total
        stats["neMean"] = ne_total / npe if npe > 0 and np.isfinite(ne_total) else np.nan
        stats["niMean"] = ni_total / npe if npe > 0 and np.isfinite(ni_total) else np.nan
        stats["gainSim"] = stats["neMean"]
        return stats
    except Exception:
        return stats


def _alpha_from_gain_and_gap(gain, gap_mm):
    gain = _safe_float(gain)
    gap_mm = _safe_float(gap_mm)
    if not np.isfinite(gain) or not np.isfinite(gap_mm) or gain <= 0.0 or gap_mm <= 0.0:
        return np.nan
    gap_cm = gap_mm * 0.1
    return float(np.log(gain) / gap_cm)


def _alpha_ion_from_ni_and_gap(ni_mean, gap_mm):
    ni_mean = _safe_float(ni_mean)
    gap_mm = _safe_float(gap_mm)
    if not np.isfinite(ni_mean) or not np.isfinite(gap_mm) or gap_mm <= 0.0:
        return np.nan
    return float(ni_mean / (gap_mm * 0.1))


def load_gas_dataframe_from_roots(
    folder,
    tree_name="dataOfGas",
    recursive=True,
    min_npe_for_alpha=DEFAULT_MIN_NPE_FOR_ALPHA,
):
    """
    Recorre todos los .root de una carpeta y devuelve un DataFrame con la
    información de gas y de avalancha.

    alphaEff se calcula desde la simulación:
        alphaEff = ln(<ne>) / gap_cm

    Solo se marca como válida para ajuste si npe >= min_npe_for_alpha.
    """
    pattern = os.path.join(folder, "**", "*.root") if recursive else os.path.join(folder, "*.root")
    root_files = sorted(glob.glob(pattern, recursive=recursive))

    rows = []

    for filepath in root_files:
        try:
            with uproot.open(filepath) as f:
                if tree_name not in f:
                    continue

                tree = f[tree_name]
                fallback = _fallback_avalanche_stats(f)

                gas1_arr = _read_first_existing_branch(tree, ["gas1"])
                gas2_arr = _read_first_existing_branch(tree, ["gas2"])
                comp1_arr = _read_first_existing_branch(tree, ["composition1"])
                comp2_arr = _read_first_existing_branch(tree, ["composition2"])
                efield_arr = _read_first_existing_branch(tree, ["electricField"])
                gap_arr = _read_first_existing_branch(tree, ["gap_mm", "gap"])
                pressure_arr = _read_first_existing_branch(tree, ["pressure"])
                pressure_bar_arr = _read_first_existing_branch(tree, ["pressureBar", "pressure_bar"])
                temp_arr = _read_first_existing_branch(tree, ["temp", "temperature"])
                vz_arr = _read_first_existing_branch(tree, ["driftVelocity", "vz"])

                npe_arr = _read_first_existing_branch(tree, ["npe"])
                ne_total_arr = _read_first_existing_branch(tree, ["neTotal", "ne_total"])
                ni_total_arr = _read_first_existing_branch(tree, ["niTotal", "ni_total"])
                ne_mean_arr = _read_first_existing_branch(tree, ["neMean", "ne_mean"])
                ni_mean_arr = _read_first_existing_branch(tree, ["niMean", "ni_mean"])
                gain_arr = _read_first_existing_branch(tree, ["gainSim", "gain_sim", "gainTeo"])
                alpha_arr = _read_first_existing_branch(tree, ["alphaEff", "alphaFromNe"])
                alpha_from_ne_arr = _read_first_existing_branch(tree, ["alphaFromNe"])
                alpha_from_ni_arr = _read_first_existing_branch(tree, ["alphaFromNi"])
                alpha_source_arr = _read_first_existing_branch(tree, ["alphaSource"])

                n = tree.num_entries

                for i in range(n):
                    gap_i = _safe_float(_value_at(gap_arr, i))
                    pressure_i = _safe_float(_value_at(pressure_arr, i))
                    pressure_bar_i = _safe_float(_value_at(pressure_bar_arr, i))
                    if not np.isfinite(pressure_bar_i) and np.isfinite(pressure_i):
                        pressure_bar_i = pressure_i / TORR_PER_BAR

                    npe_i = _safe_int(_value_at(npe_arr, i, fallback["npe"]), default=_safe_int(fallback["npe"], default=0))
                    ne_total_i = _safe_float(_value_at(ne_total_arr, i, fallback["neTotal"]))
                    ni_total_i = _safe_float(_value_at(ni_total_arr, i, fallback["niTotal"]))
                    ne_mean_i = _safe_float(_value_at(ne_mean_arr, i, fallback["neMean"]))
                    ni_mean_i = _safe_float(_value_at(ni_mean_arr, i, fallback["niMean"]))
                    gain_i = _safe_float(_value_at(gain_arr, i, fallback["gainSim"]))

                    if not np.isfinite(ne_mean_i) and npe_i > 0 and np.isfinite(ne_total_i):
                        ne_mean_i = ne_total_i / npe_i
                    if not np.isfinite(ni_mean_i) and npe_i > 0 and np.isfinite(ni_total_i):
                        ni_mean_i = ni_total_i / npe_i
                    if not np.isfinite(gain_i):
                        gain_i = ne_mean_i

                    alpha_from_ne_i = _safe_float(_value_at(alpha_from_ne_arr, i))
                    alpha_from_ni_i = _safe_float(_value_at(alpha_from_ni_arr, i))
                    # Aunque el ROOT antiguo tenga una rama alphaEff calculada por
                    # Magboltz, la ignoramos: alpha se reconstruye siempre desde
                    # la avalancha simulada.
                    _ = _safe_float(_value_at(alpha_arr, i))

                    if not np.isfinite(alpha_from_ne_i):
                        alpha_from_ne_i = _alpha_from_gain_and_gap(gain_i, gap_i)
                    if not np.isfinite(alpha_from_ni_i):
                        alpha_from_ni_i = _alpha_ion_from_ni_and_gap(ni_mean_i, gap_i)
                    alpha_eff_i = alpha_from_ne_i

                    valid_for_alpha = bool(
                        npe_i >= int(min_npe_for_alpha)
                        and np.isfinite(alpha_eff_i)
                        and alpha_eff_i > 0.0
                    )
                    if not valid_for_alpha:
                        alpha_eff_i = np.nan

                    alpha_source = _value_at(alpha_source_arr, i, "simulation_npe")
                    if alpha_source is None or (isinstance(alpha_source, float) and pd.isna(alpha_source)):
                        alpha_source = "simulation_npe"

                    rows.append({
                        "file": filepath,
                        "sourceFiles": filepath,
                        "nRuns": 1,
                        "gas1": _value_at(gas1_arr, i, None),
                        "gas2": _value_at(gas2_arr, i, None),
                        "composition1": _safe_float(_value_at(comp1_arr, i)),
                        "composition2": _safe_float(_value_at(comp2_arr, i)),
                        "electricField": _safe_float(_value_at(efield_arr, i)),
                        "gap": gap_i,
                        "pressure": pressure_i,
                        "pressureBar": pressure_bar_i,
                        "temperature": _safe_float(_value_at(temp_arr, i)),
                        "npe": npe_i,
                        "neTotal": ne_total_i,
                        "niTotal": ni_total_i,
                        "neMean": ne_mean_i,
                        "niMean": ni_mean_i,
                        "gainSim": gain_i,
                        "alphaEff": alpha_eff_i,
                        "alphaFromNe": alpha_from_ne_i if valid_for_alpha else np.nan,
                        "alphaFromNi": alpha_from_ni_i if valid_for_alpha else np.nan,
                        "vz": _safe_float(_value_at(vz_arr, i)),
                        "validForAlpha": valid_for_alpha,
                        "alphaSource": alpha_source,
                    })

        except Exception as e:
            print(f"[WARNING] No se pudo leer {filepath}: {e}")

    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def _normalize_dataframe_for_merge(df):
    """Normaliza tipos para comparar filas sin problemas raros de dtype."""
    df = df.copy()

    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    text_cols = ["file", "sourceFiles", "gas1", "gas2", "alphaSource"]
    numeric_cols = [
        "nRuns", "composition1", "composition2", "electricField",
        "gap", "pressure", "pressureBar", "temperature", "npe",
        "neTotal", "niTotal", "neMean", "niMean", "gainSim",
        "alphaEff", "alphaFromNe", "alphaFromNi", "vz",
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "validForAlpha" in df.columns:
        df["validForAlpha"] = df["validForAlpha"].fillna(False).astype(bool)

    return df[CSV_COLUMNS]


def _is_close(a, b, tol=1e-9):
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(float(a) - float(b)) < tol


def _is_pure_gas_row(row):
    """
    Considera gas puro si:
    - composition1 es 100 o 1
    - y composition2 es 0 o NaN
    """
    c1 = row.get("composition1", np.nan)
    c2 = row.get("composition2", np.nan)

    pure_c1 = _is_close(c1, 100.0) or _is_close(c1, 1.0)
    pure_c2 = pd.isna(c2) or _is_close(c2, 0.0)

    return pure_c1 and pure_c2


def _build_canonical_dedup_columns(df):
    """
    Construye columnas auxiliares para agrupar.
    Si el gas es puro, gas2/composition2 no cuentan y composition1 se normaliza a 100.
    """
    df = df.copy()

    canon_gas1 = []
    canon_gas2 = []
    canon_comp1 = []
    canon_comp2 = []

    for _, row in df.iterrows():
        g1 = row.get("gas1", pd.NA)
        g2 = row.get("gas2", pd.NA)
        c1 = row.get("composition1", np.nan)
        c2 = row.get("composition2", np.nan)

        if _is_pure_gas_row(row):
            canon_gas1.append(g1)
            canon_gas2.append(pd.NA)
            canon_comp1.append(100.0)
            canon_comp2.append(0.0)
        else:
            pairs = []
            if not pd.isna(g1) and not pd.isna(c1) and not _is_close(c1, 0.0):
                pairs.append((str(g1), float(c1)))
            if not pd.isna(g2) and not pd.isna(c2) and not _is_close(c2, 0.0):
                pairs.append((str(g2), float(c2)))
            pairs = sorted(pairs, key=lambda item: item[0])

            if len(pairs) > 0:
                canon_gas1.append(pairs[0][0])
                canon_comp1.append(pairs[0][1])
            else:
                canon_gas1.append(pd.NA)
                canon_comp1.append(np.nan)

            if len(pairs) > 1:
                canon_gas2.append(pairs[1][0])
                canon_comp2.append(pairs[1][1])
            else:
                canon_gas2.append(pd.NA)
                canon_comp2.append(0.0)

    df["_canon_gas1"] = canon_gas1
    df["_canon_gas2"] = canon_gas2
    df["_canon_composition1"] = canon_comp1
    df["_canon_composition2"] = canon_comp2

    return df


def _add_group_keys(df):
    df = _build_canonical_dedup_columns(df)
    for col in [
        "_canon_composition1", "_canon_composition2",
        "electricField", "gap", "pressure", "temperature",
    ]:
        df[f"_key_{col}"] = pd.to_numeric(df[col], errors="coerce").round(8)

    df["_key_gas1"] = df["_canon_gas1"].astype("string").fillna("<NA>")
    df["_key_gas2"] = df["_canon_gas2"].astype("string").fillna("<NA>")

    return df


def _group_key_columns():
    return [
        "_key_gas1", "_key_gas2",
        "_key__canon_composition1", "_key__canon_composition2",
        "_key_electricField", "_key_gap", "_key_pressure", "_key_temperature",
    ]


def _weighted_average(values, weights):
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def _collect_sources(group):
    sources = []
    for col in ["sourceFiles", "file"]:
        if col not in group.columns:
            continue
        for value in group[col].dropna().astype(str):
            for part in value.split(";"):
                part = part.strip()
                if part and part not in sources:
                    sources.append(part)
    return ";".join(sources)


def aggregate_matching_simulations(df, min_npe_for_alpha=DEFAULT_MIN_NPE_FOR_ALPHA):
    """
    Agrega filas con la misma mezcla, presión, gap y campo eléctrico.

    La media de ne/ni/gain se pondera por npe. Después alphaEff se recalcula
    desde la ganancia media combinada, de modo que un ROOT con más npe pesa más.
    """
    df = _normalize_dataframe_for_merge(df)
    if df.empty:
        return df

    work = _add_group_keys(df)
    grouped = work.groupby(_group_key_columns(), dropna=False, sort=False)

    rows = []
    for _, group in grouped:
        group = group.copy()
        out = group.iloc[-1].copy()
        weights = pd.to_numeric(group["npe"], errors="coerce").fillna(0.0).clip(lower=0.0)
        total_npe = float(weights.sum())

        if total_npe > 0.0:
            out["npe"] = int(round(total_npe))
            out["neMean"] = _weighted_average(group["neMean"], weights)
            out["niMean"] = _weighted_average(group["niMean"], weights)
            out["gainSim"] = _weighted_average(group["gainSim"], weights)

            ne_total = pd.to_numeric(group["neTotal"], errors="coerce")
            ni_total = pd.to_numeric(group["niTotal"], errors="coerce")
            out["neTotal"] = float(ne_total.sum(skipna=True)) if ne_total.notna().any() else out["neMean"] * total_npe
            out["niTotal"] = float(ni_total.sum(skipna=True)) if ni_total.notna().any() else out["niMean"] * total_npe
        else:
            out["npe"] = np.nan

        out["nRuns"] = int(pd.to_numeric(group["nRuns"], errors="coerce").fillna(1).sum())
        out["sourceFiles"] = _collect_sources(group)
        out["file"] = group["file"].dropna().astype(str).iloc[-1] if group["file"].notna().any() else pd.NA
        out["alphaSource"] = "simulation_npe_weighted"

        alpha_from_ne = _alpha_from_gain_and_gap(out.get("gainSim", np.nan), out.get("gap", np.nan))
        alpha_from_ni = _alpha_ion_from_ni_and_gap(out.get("niMean", np.nan), out.get("gap", np.nan))
        valid = bool(
            _safe_int(out.get("npe", np.nan)) >= int(min_npe_for_alpha)
            and np.isfinite(alpha_from_ne)
            and alpha_from_ne > 0.0
        )

        out["validForAlpha"] = valid
        out["alphaEff"] = alpha_from_ne if valid else np.nan
        out["alphaFromNe"] = alpha_from_ne if valid else np.nan
        out["alphaFromNi"] = alpha_from_ni if valid else np.nan

        rows.append(out[CSV_COLUMNS])

    result = pd.DataFrame(rows, columns=CSV_COLUMNS)
    result = result.sort_values(
        by=["gas1", "gas2", "composition1", "composition2", "pressure", "gap", "electricField"],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    return result


def _key_set(df):
    if df.empty:
        return set()
    work = _add_group_keys(_normalize_dataframe_for_merge(df))
    keys = set()
    for _, row in work.iterrows():
        keys.add(tuple(row[col] for col in _group_key_columns()))
    return keys


def _drop_keys(df, keys_to_drop):
    if df.empty or not keys_to_drop:
        return df
    work = _add_group_keys(_normalize_dataframe_for_merge(df))
    keep = []
    for _, row in work.iterrows():
        key = tuple(row[col] for col in _group_key_columns())
        keep.append(key not in keys_to_drop)
    return work.loc[keep, CSV_COLUMNS].reset_index(drop=True)


def merge_with_existing_csv(df_new, output_csv, min_npe_for_alpha=DEFAULT_MIN_NPE_FOR_ALPHA):
    """
    Mantiene información antigua si no hay ROOT equivalente en la carpeta actual.
    Para claves presentes en los ROOT actuales, reemplaza la fila antigua por la
    agregación ponderada de los ROOT encontrados. Esto evita contar dos veces el
    mismo backup cada vez que se actualiza el CSV.
    """
    if os.path.exists(output_csv):
        try:
            df_old = pd.read_csv(output_csv)
            print(f"[INFO] CSV existente encontrado: {output_csv}")
            print(f"[INFO] Filas previas: {len(df_old)}")
        except Exception as e:
            print(f"[WARNING] No se pudo leer el CSV existente ({output_csv}): {e}")
            df_old = pd.DataFrame(columns=CSV_COLUMNS)
    else:
        df_old = pd.DataFrame(columns=CSV_COLUMNS)

    df_old = _normalize_dataframe_for_merge(df_old)
    df_new = _normalize_dataframe_for_merge(df_new)
    df_new_agg = aggregate_matching_simulations(df_new, min_npe_for_alpha=min_npe_for_alpha)

    if df_new_agg.empty:
        df_final = df_old
    else:
        new_keys = _key_set(df_new_agg)
        df_old_keep = _drop_keys(df_old, new_keys)
        df_final = pd.concat([df_old_keep, df_new_agg], ignore_index=True)

    df_final = _normalize_dataframe_for_merge(df_final)

    print(f"[INFO] Filas nuevas leídas de ROOT: {len(df_new)}")
    print(f"[INFO] Filas nuevas agregadas: {len(df_new_agg)}")
    print(f"[INFO] Filas finales: {len(df_final)}")

    return df_final


def export_roots_to_csv(
    folder,
    output_csv="gas_data.csv",
    tree_name="dataOfGas",
    recursive=True,
    min_npe_for_alpha=DEFAULT_MIN_NPE_FOR_ALPHA,
):
    """
    Actualiza el CSV con lo leído de los ROOT, sin perder información vieja si
    desaparece algún archivo. Las alphas válidas salen exclusivamente de
    simulaciones con npe >= min_npe_for_alpha.
    """
    df_new = load_gas_dataframe_from_roots(
        folder,
        tree_name=tree_name,
        recursive=recursive,
        min_npe_for_alpha=min_npe_for_alpha,
    )
    df_final = merge_with_existing_csv(
        df_new,
        output_csv,
        min_npe_for_alpha=min_npe_for_alpha,
    )
    df_final.to_csv(output_csv, index=False)

    print(f"CSV guardado en: {output_csv}")
    print(f"Filas totales en CSV: {len(df_final)}")

    return df_final
