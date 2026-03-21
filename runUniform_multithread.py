#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 13:32:27 2025

@author: pablo
"""
import subprocess
import os
from multiprocessing import Pool, cpu_count
from functools import partial

def run_uniformE(args, is_isotropic):
    mode = args[0]

    if mode==0: 
        """Ejecuta ./uniformE con la lista de argumentos en el directorio build."""
        dir_output = "build"
        print(f"--> Lanzando simulación:\n    ./uniformE {' '.join(args[1:])}")
        subprocess.run(["./uniformE"] + args, cwd=dir_output)
        print(f"--> Finalizó simulación:\n    ./uniformE {' '.join(args[1:])}")

    elif mode==1:
        """Ejecuta ./uniformE con la lista de argumentos en el directorio build."""
        dir_output = "build"

        subprocess.run(["./uniformE"] + args, cwd=dir_output)

        print(f"--> Lanzando simulación:\n    ./uniformE {' '.join(args[1:])}")
        subprocess.run(["./uniformE"] + args, cwd=dir_output)
        print(f"--> Finalizó simulación:\n    ./uniformE {' '.join(args[1:])}")

    elif mode==2:
        """Ejecuta ./uniformE con la lista de argumentos en el directorio build."""
        dir_output = "build"

        subprocess.run(["./uniformE"] + args, cwd=dir_output)

        print(f"--> Lanzando simulación:\n    ./uniformE {' '.join(args[1:])}")
        subprocess.run(["./uniformE"] + args, cwd=dir_output)
        print(f"--> Finalizó simulación:\n    ./uniformE {' '.join(args[1:])}")

    else:
        print("ELIJE UN MODO VALIDO")


######################################################################
# Parámetros del usuario

# Los parámetros de simulación van en listas. Con n controlamos cuantos elementos de la lista
# se ejecutan partiendo del elemento 0. Es decir, si es n=1, solo se ejecutará el primer ele-
# mento de todas las listas.

n = 4

######################################################################
# Modo de la simulación

# En función del modo de la simulación se corre de una manera u otra. Para elegir el modo se 
# elige un número, en función del número se activa una u otra.

#   -> -1: Mode isotropic (no implemented yet).
#   ->  0: Mode gap fixed. Modo clásico.
#   ->  1: Mode gain fixed gap. Dado una ganancia se calcula el gap.
#   ->  2: Mode gain fixed time. Dada una ganancia se calcula el tiempo aceptado para e- generados.


mode = [0] * n


######################################################################
# Parametros de la simulación


npe       = [10000] * n            # e- primarios

pressure  = [1] * n               # bar

gas1      = ["ar"] * n   

mixture1  = [95., 90., 33., 0.]           # % gas

gas2      = ["cf4"] * n

mixture2  = [5., 10., 67., 100.]                 # % gas

fieldE    = [65000, 78000, 88000, 95000]   # V/cm, is_isotropic)

height = [0.98] * n         # Decide a la altura que quieras lanzar la simulación (0 = Pegado al catodo +, 1 = Pegado al anodo -)


######################################################################
# Aqui eliges la ganancia o gap requeridos

#  -> Si eliges mode gap, se lanza con el gap querido.

#  -> Si eliges mode gain, se calcula el gap para gas, ganancia, campo eléctrico y presión demandadas. 

gap       = [0.05] * n               # mm

gain      = [10e4] * n               # e⁻/e⁻p


######################################################################


# ---------------------------
# COMPILACIÓN ÚNICA
# ---------------------------

subprocess.run(["rm", "-rf", "build/"])
os.makedirs("build", exist_ok=True)
subprocess.run(["cmake", ".."], cwd="build")
subprocess.run("make -j$Nproc", shell=True, cwd="build")

root_dir = "rootArchives"
os.makedirs("rootArchives", exist_ok=True)

# ---------------------------
# PREPARACIÓN DE ARGUMENTOS
# ---------------------------

all_jobs = []

for i in range(n):
    rootFileName = (
        f"../{root_dir}/"
        f"{gas1[i]}_{mixture2[i]:.1f}{gas2[i]}_"
        f"{fieldE[i]/1000:.1f}kVcm_"
        f"{pressure[i]}bar_"
        f"{gap[i]:.2f}mm_{npe[i]}npe.root"
    )
    
    args = [
        mode[i],
        rootFileName,
        str(fieldE[i]),
        f"{gap[i]:.2f}",
        str(pressure[i]),
        str(npe[i]),
        gas1[i],
        f"{mixture1[i]:.3f}",
        gas2[i],
        f"{mixture2[i]:.3f}",
        str(height[i])
    ]

    all_jobs.append(args)

# ---------------------------
# MULTIPROCESSING
# ---------------------------

num_cores = min(cpu_count(), n)
print(f"\nUsando {num_cores} núcleos para las simulaciones...\n")

with Pool(processes=num_cores) as pool:
    pool.map(partial(run_uniformE), all_jobs)

print("\n✔ Todas las simulaciones han terminado.\n")
