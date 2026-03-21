
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
from multiprocessing import Pool, cpu_count

from gainCalculation import (
    update_csv_and_fit_mix,
    required_gap_for_gain,
    required_E_for_gain,
    pressure_bar_to_torr,
    mix_slug,
)


def run_uniformE(args):
    """
    Ejecuta ./uniformE con la lista de argumentos en el directorio build.
    args debe contener SOLO strings.
    """
    dir_output = "build"
    printable = " ".join(args)
    print(f"--> Lanzando simulación:\n    ./uniformE {printable}")
    subprocess.run(["./uniformE"] + args, cwd=dir_output, check=True)
    print(f"--> Finalizó simulación:\n    ./uniformE {printable}")


######################################################################
# Parámetros del usuario

n = 2

######################################################################
# Modo de la simulación
#
#   -> 0: Mode gap fixed. Modo clásico.
#   -> 1: Mode gain fixed gap. Dada una ganancia se calcula el gap.
#   -> 2: Mode gain fixed field. Dada una ganancia se calcula el campo eléctrico.
#
# En modos 1 y 2:
#   - se actualiza el CSV desde rootArchives
#   - se ajusta alpha/p = A exp(-B p/E) para la mezcla pedida
#   - opcionalmente se guarda un PDF del ajuste alpha vs E
######################################################################

mode = [1] * n

######################################################################
# Parametros de la simulación

npe = [100] * n
pressure = [1] * n            # bar
gas1 = ["ar"] * n              # "C2H2F4"
mixture1 = [100] * n   # %
gas2 = ["cf4"] * n
mixture2 = [0] * n    # %
fieldE = [50000, 70000]  # V/cm
height = [0.98] * n
printTable = [1] * n # 0 = True, 1 = False

######################################################################
# Aqui eliges la ganancia o gap requeridos
#
# mode 0 -> usa gap[i]
# mode 1 -> usa gain[i] y calcula gap[i]
# mode 2 -> usa gain[i] y calcula fieldE[i]
######################################################################

gap = [0.05] * n               # mm
gain = [1.0e3] * n             # e-/e-p

######################################################################
# Configuración del ajuste / CSV / PDFs

root_dir = "rootArchives"
csv_database = "gas_data.csv"
fit_pdf_dir = "fitPlots"
save_fit_pdf = True

######################################################################
# COMPILACIÓN ÚNICA

subprocess.run(["rm", "-rf", "build/"])
os.makedirs("build", exist_ok=True)
subprocess.run(["cmake", ".."], cwd="build", check=True)
subprocess.run("make -j$(nproc)", shell=True, cwd="build", check=True)

os.makedirs(root_dir, exist_ok=True)
os.makedirs(fit_pdf_dir, exist_ok=True)

######################################################################
# PREPARACIÓN DE ARGUMENTOS

all_jobs = []

for i in range(n):
    gas1_i = gas1[i]
    gas2_i = gas2[i]
    mixture1_i = float(mixture1[i])
    mixture2_i = float(mixture2[i])
    pressure_bar_i = float(pressure[i])
    pressure_torr_i = pressure_bar_to_torr(pressure_bar_i)
    printTable_i = printTable[i]
    fieldE_i = float(fieldE[i])
    gap_i = float(gap[i])

    if mode[i] in (1, 2):
        pdf_path = None
        if save_fit_pdf:
            pdf_path = os.path.join(
                fit_pdf_dir,
                f"alpha_fit_{mix_slug(gas1_i, mixture1_i, gas2_i, mixture2_i)}.pdf"
            )

        fit_bundle = update_csv_and_fit_mix(
            root_folder=root_dir,
            csv_path=csv_database,
            gas1=gas1_i,
            composition1=mixture1_i,
            gas2=gas2_i,
            composition2=mixture2_i,
            make_pdf=save_fit_pdf,
            pdf_path=pdf_path,
        )

        fit_result = fit_bundle["fit"]

        if mode[i] == 1:
            gap_i = float(required_gap_for_gain(
                gain=gain[i],
                E=fieldE_i,
                p=pressure_torr_i,
                fit_result=fit_result,
            ))
            print(
                f"[MODE 1] Mezcla {fit_bundle['mix_label']} -> "
                f"gain={gain[i]:.5g}, E={fieldE_i:.3f} V/cm, p={pressure_bar_i:.3f} bar "
                f"=> gap calculado = {gap_i:.6f} mm"
            )

        elif mode[i] == 2:
            fieldE_i = float(required_E_for_gain(
                gain=gain[i],
                p=pressure_torr_i,
                gap_mm=gap_i,
                fit_result=fit_result,
            ))
            print(
                f"[MODE 2] Mezcla {fit_bundle['mix_label']} -> "
                f"gain={gain[i]:.5g}, gap={gap_i:.6f} mm, p={pressure_bar_i:.3f} bar "
                f"=> E calculado = {fieldE_i:.6f} V/cm"
            )

        if save_fit_pdf and fit_bundle["pdf_path"] is not None:
            print(f"[FIT PDF] Guardado en: {fit_bundle['pdf_path']}")

    rootFileName = (
        f"../{root_dir}/"
        f"{gas1_i}_{mixture1_i:.1f}_{gas2_i}_{mixture2_i:.1f}_"
        f"{fieldE_i/1000:.1f}kVcm_"
        f"{pressure_bar_i:.3f}bar_"
        f"{gap_i:.4f}mm_{int(npe[i])}npe.root"
    )

    args = [
        str(rootFileName),
        str(fieldE_i),
        f"{gap_i:.6f}",
        str(pressure_bar_i),
        str(int(npe[i])),
        str(gas1_i),
        f"{mixture1_i:.6f}",
        str(gas2_i),
        f"{mixture2_i:.6f}",
        str(height[i]),
        str(printTable_i),
    ]

    all_jobs.append(args)

######################################################################
# MULTIPROCESSING

num_cores = min(cpu_count(), n)
print(f"\nUsando {num_cores} núcleos para las simulaciones...\n")

with Pool(processes=num_cores) as pool:
    pool.map(run_uniformE, all_jobs)

print("\n✔ Todas las simulaciones han terminado.\n")
