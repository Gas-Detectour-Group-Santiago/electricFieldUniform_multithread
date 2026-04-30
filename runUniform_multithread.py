#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import threading
import numpy as np

from tqdm import tqdm

from gainCalculation import (
    update_csv_and_fit_mix,
    required_gap_for_gain,
    required_E_for_gain,
    pressure_bar_to_torr,
    mix_slug,
)

######################################################################
# Parámetros del usuario

n = 12

######################################################################
# Modo de la simulación
#
#   0: gap fijo
#   1: gain fija -> calcula gap
#   2: gain fija -> calcula campo E
#
######################################################################

mode = [2] * n

######################################################################
# Parámetros de la simulación

# npe             = [50] * n
# pressure        = [5] * n                                        # bar
# gas1            = ["ar"] * (n-1) + ["cf4"]
# mixture1        = [99.9,99.5,99,98,95,90,80,50,100] * n        # %
# gas2            = ["cf4"] * (n-1) + ["ar"]
# mixture2        = [0.1,0.5,1,2,5,10,20,50,0] * n           # %
# fieldE          = [1] * n                                  # V/cm
# height          = [1.5] * n
# printTable      = [1] * n        # 0 -> True       

npe             = [100] * n
pressure        = [10] * n                                        # bar
gas1            = ["ar"] * n
mixture1        = [99] * n        # %
gas2            = ["cf4"] * n
mixture2        = [1] * n          # %
fieldE          = np.linspace(10,120,n)*1000                                 # V/cm
height          = [1.5] * n
printTable      = [1] * n        # 0 -> True                                 

######################################################################
#
# mode 0 -> usa gap[i]
# mode 1 -> usa gain[i] y calcula gap[i]
# mode 2 -> usa gain[i] y calcula fieldE[i]
#
######################################################################

gap = [0.57] * n                  # mm
gain = [1.0e4] * n                 # e-/e-p

######################################################################
# Configuración del ajuste / CSV / PDFs

root_dir = "rootArchives"
csv_database = "gas_data.csv"
fit_pdf_dir = "fitPlots"
save_fit_pdf = True

######################################################################
# Configuración de ejecución

build_dir = "build"
exe_name = "./uniformE"

# Lock para que tqdm no se corrompa al actualizar desde varios threads.
tqdm_lock = threading.Lock()


def compile_project():
    subprocess.run(["rm", "-rf", build_dir], check=True)
    os.makedirs(build_dir, exist_ok=True)

    subprocess.run(["cmake", ".."], cwd=build_dir, check=True)
    subprocess.run("make -j$(nproc)", shell=True, cwd=build_dir, check=True)

    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(fit_pdf_dir, exist_ok=True)


def build_jobs():
    all_jobs = []

    for i in range(n):
        gas1_i = gas1[i]
        gas2_i = gas2[i]
        mixture1_i = float(mixture1[i])
        mixture2_i = float(mixture2[i])
        pressure_bar_i = float(pressure[i])
        pressure_torr_i = pressure_bar_to_torr(pressure_bar_i)
        printTable_i = int(printTable[i])
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
                gap_i = float(
                    required_gap_for_gain(
                        gain=gain[i],
                        E=fieldE_i,
                        p=pressure_torr_i,
                        fit_result=fit_result,
                    )
                )
                print(
                    f"[MODE 1] Mezcla {fit_bundle['mix_label']} -> "
                    f"gain={gain[i]:.5g}, E={fieldE_i:.3f} V/cm, "
                    f"p={pressure_bar_i:.3f} bar => gap calculado = {gap_i:.6f} mm"
                )

            elif mode[i] == 2:
                fieldE_i = float(
                    required_E_for_gain(
                        gain=gain[i],
                        p=pressure_torr_i,
                        gap_mm=gap_i,
                        fit_result=fit_result,
                    )
                )
                print(
                    f"[MODE 2] Mezcla {fit_bundle['mix_label']} -> "
                    f"gain={gain[i]:.5g}, gap={gap_i:.6f} mm, "
                    f"p={pressure_bar_i:.3f} bar => E calculado = {fieldE_i:.6f} V/cm"
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

        # OJO:
        # el último argumento es jobId
        args = [
            str(rootFileName),
            f"{fieldE_i:.6f}",
            f"{gap_i:.6f}",
            str(pressure_bar_i),
            str(int(npe[i])),
            str(gas1_i),
            f"{mixture1_i:.6f}",
            str(gas2_i),
            f"{mixture2_i:.6f}",
            f"{height[i]:.4f}",
            str(printTable_i),
            str(i),  # jobId
        ]

        all_jobs.append(args)

    return all_jobs
def monitor_process(proc, bar, job_index):
    while True:
        raw = proc.stdout.readline()
        if not raw:
            break

        line = raw.decode("utf-8", errors="ignore").strip()
        if not line:
            continue

        if line.startswith("PROGRESS"):
            parts = line.split()
            if len(parts) != 4:
                continue

            try:
                _, job_id_str, current_str, total_str = parts
                job_id = int(job_id_str)
                current = int(current_str)
                total = int(total_str)
            except ValueError:
                continue

            if job_id != job_index:
                continue

            with tqdm_lock:
                if bar.total != total:
                    bar.total = total
                bar.n = current
                bar.refresh()

        elif line.startswith("DONE"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    done_job = int(parts[1])
                except ValueError:
                    done_job = None

                if done_job == job_index:
                    with tqdm_lock:
                        bar.n = bar.total
                        bar.refresh()

    return_code = proc.wait()

    with tqdm_lock:
        if return_code == 0:
            bar.n = bar.total
            bar.refresh()
        else:
            # NO imprimir aquí mientras las barras siguen activas
            pass
        
def launch_jobs(all_jobs):
    procs = []
    threads = []
    bars = []

    print(f"\nLanzando {len(all_jobs)} simulaciones en paralelo...\n")

    for i, args in enumerate(all_jobs):
        total_events = int(args[4])

        bar = tqdm(
            total=total_events,
            desc=f"job {i}",
            position=i,
            leave=True,
            dynamic_ncols=True,
        )
        bars.append(bar)

        proc = subprocess.Popen(
            [exe_name] + args,
            cwd=build_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
        procs.append(proc)

        th = threading.Thread(
            target=monitor_process,
            args=(proc, bar, i),
            daemon=True,
        )
        th.start()
        threads.append(th)

    for th in threads:
        th.join()

    # Refresco final y cierre SOLO cuando ya terminó todo
    with tqdm_lock:
        for bar in bars:
            bar.n = bar.total
            bar.refresh()

        print()  # baja una línea una sola vez al final

        for bar in bars:
            bar.close()

    print("✔ Todas las simulaciones han terminado.\n")

def main():
    compile_project()
    all_jobs = build_jobs()
    launch_jobs(all_jobs)


if __name__ == "__main__":
    main()