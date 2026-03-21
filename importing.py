
import os
import glob
import numpy as np
import pandas as pd
import uproot


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


def load_gas_dataframe_from_roots(folder, tree_name="dataOfGas", recursive=True):
    """
    Recorre todos los .root de una carpeta y devuelve un DataFrame con:
    gas1, gas2, composition1, composition2, electricField, gap, pressure,
    temperature, alpha, vz

    Añade además una columna 'file' con la ruta del archivo.
    """
    pattern = os.path.join(folder, "**", "*.root") if recursive else os.path.join(folder, "*.root")
    root_files = glob.glob(pattern, recursive=recursive)

    rows = []

    for filepath in root_files:
        try:
            with uproot.open(filepath) as f:
                if tree_name not in f:
                    continue

                tree = f[tree_name]

                gas1_arr = _read_first_existing_branch(tree, ["gas1"])
                gas2_arr = _read_first_existing_branch(tree, ["gas2"])
                comp1_arr = _read_first_existing_branch(tree, ["composition1"])
                comp2_arr = _read_first_existing_branch(tree, ["composition2"])
                efield_arr = _read_first_existing_branch(tree, ["electricField"])
                gap_arr = _read_first_existing_branch(tree, ["gap_mm", "gap"])
                pressure_arr = _read_first_existing_branch(tree, ["pressure"])
                temp_arr = _read_first_existing_branch(tree, ["temp", "temperature"])
                alpha_arr = _read_first_existing_branch(tree, ["alpha"])
                vz_arr = _read_first_existing_branch(tree, ["driftVelocity", "vz"])

                n = tree.num_entries

                for i in range(n):
                    rows.append({
                        "file": filepath,
                        "gas1": _to_python_scalar(gas1_arr[i]) if gas1_arr is not None else None,
                        "gas2": _to_python_scalar(gas2_arr[i]) if gas2_arr is not None else None,
                        "composition1": _to_python_scalar(comp1_arr[i]) if comp1_arr is not None else np.nan,
                        "composition2": _to_python_scalar(comp2_arr[i]) if comp2_arr is not None else np.nan,
                        "electricField": _to_python_scalar(efield_arr[i]) if efield_arr is not None else np.nan,
                        "gap": _to_python_scalar(gap_arr[i]) if gap_arr is not None else np.nan,
                        "pressure": _to_python_scalar(pressure_arr[i]) if pressure_arr is not None else np.nan,
                        "temperature": _to_python_scalar(temp_arr[i]) if temp_arr is not None else np.nan,
                        "alpha": _to_python_scalar(alpha_arr[i]) if alpha_arr is not None else np.nan,
                        "vz": _to_python_scalar(vz_arr[i]) if vz_arr is not None else np.nan,
                    })

        except Exception as e:
            print(f"[WARNING] No se pudo leer {filepath}: {e}")

    df = pd.DataFrame(rows, columns=[
        "file",
        "gas1", "gas2",
        "composition1", "composition2",
        "electricField", "gap", "pressure", "temperature",
        "alpha", "vz"
    ])

    return df


def _normalize_dataframe_for_merge(df):
    """Normaliza tipos para comparar filas sin problemas raros de dtype."""
    if df.empty:
        return df.copy()

    df = df.copy()

    text_cols = ["file", "gas1", "gas2"]
    numeric_cols = ["composition1", "composition2", "electricField",
                    "gap", "pressure", "temperature", "alpha", "vz"]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


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
    Construye columnas auxiliares para deduplicar.
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
            canon_gas1.append(g1)
            canon_gas2.append(g2)
            canon_comp1.append(c1)
            canon_comp2.append(c2)

    df["_canon_gas1"] = canon_gas1
    df["_canon_gas2"] = canon_gas2
    df["_canon_composition1"] = canon_comp1
    df["_canon_composition2"] = canon_comp2

    return df
def merge_with_existing_csv(df_new, output_csv):
    """
    Mantiene la información antigua del CSV aunque ya no exista el ROOT.
    Si un análisis nuevo tiene las mismas condiciones de entrada que uno viejo,
    se queda el nuevo (keep='last').
    Para gases puros, ignora gas2/composition2 y normaliza composition1 a 100.
    """
    if os.path.exists(output_csv):
        try:
            df_old = pd.read_csv(output_csv)
            print(f"[INFO] CSV existente encontrado: {output_csv}")
            print(f"[INFO] Filas previas: {len(df_old)}")
        except Exception as e:
            print(f"[WARNING] No se pudo leer el CSV existente ({output_csv}): {e}")
            df_old = pd.DataFrame(columns=df_new.columns)
    else:
        df_old = pd.DataFrame(columns=df_new.columns)

    all_columns = [
        "file",
        "gas1", "gas2",
        "composition1", "composition2",
        "electricField", "gap", "pressure", "temperature",
        "alpha", "vz"
    ]

    for col in all_columns:
        if col not in df_old.columns:
            df_old[col] = np.nan
        if col not in df_new.columns:
            df_new[col] = np.nan

    df_old = df_old[all_columns]
    df_new = df_new[all_columns]

    df_old = _normalize_dataframe_for_merge(df_old)
    df_new = _normalize_dataframe_for_merge(df_new)

    # Primero viejo, luego nuevo: así keep='last' conserva el análisis más reciente
    df_combined = pd.concat([df_old, df_new], ignore_index=True)

    # Canonizamos mezcla para tratar bien gases puros
    df_combined = _build_canonical_dedup_columns(df_combined)

    # IMPORTANTE:
    # La clave de duplicado debe usar SOLO condiciones de entrada
    dedup_cols = [
        "_canon_gas1", "_canon_gas2",
        "_canon_composition1", "_canon_composition2",
        "electricField", "gap", "pressure", "temperature"
    ]

    before = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=dedup_cols, keep="last")
    after = len(df_combined)

    df_combined = df_combined.drop(columns=[
        "_canon_gas1", "_canon_gas2",
        "_canon_composition1", "_canon_composition2"
    ])

    print(f"[INFO] Filas combinadas antes de eliminar duplicados: {before}")
    print(f"[INFO] Filas finales: {after}")
    print(f"[INFO] Duplicados eliminados: {before - after}")

    return df_combined


def export_roots_to_csv(folder, output_csv="gas_data.csv", tree_name="dataOfGas", recursive=True):
    """
    Actualiza el CSV con lo leído de los ROOT, sin perder información vieja si
    desaparece algún archivo. Devuelve el DataFrame final.
    """
    df_new = load_gas_dataframe_from_roots(folder, tree_name=tree_name, recursive=recursive)
    df_final = merge_with_existing_csv(df_new, output_csv)
    df_final.to_csv(output_csv, index=False)

    print(f"CSV guardado en: {output_csv}")
    print(f"Filas nuevas leídas de ROOT: {len(df_new)}")
    print(f"Filas totales en CSV: {len(df_final)}")

    return df_final
