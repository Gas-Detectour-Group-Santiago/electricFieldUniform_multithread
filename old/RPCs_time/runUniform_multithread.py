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

def run_fatGemC(args, is_isotropic):
    #print("="*65, "\n" ,is_isotropic, "="*65)
    if is_isotropic: 
        """Ejecuta ./uniformE con la lista de argumentos en el directorio build."""
        dir_output = "build"
        print(f"--> Lanzando simulación:\n    ./uniformE_isotropic {' '.join(args)}")
        subprocess.run(["./uniformE_isotropic"] + args, cwd=dir_output)
        print(f"--> Finalizó simulación:\n    ./uniformE_isotropic {' '.join(args)}")
    else:
        """Ejecuta ./uniformE con la lista de argumentos en el directorio build."""
        dir_output = "build"
        print(f"--> Lanzando simulación:\n    ./uniformE {' '.join(args)}")
        subprocess.run(["./uniformE"] + args, cwd=dir_output)
        print(f"--> Finalizó simulación:\n    ./uniformE {' '.join(args)}")



######################################################################
# Parámetros del usuario

# Los parámetros de simulación van en listas. Con n controlamos cuantos elementos de la lista
#   se ejecutan partiendo del elemento 0. Es decir, si es n=1, solo se ejecutará el primer e-
#   lemento de todas las listas.

n = 2

######################################################################
# Parametros de la simulación


is_isotropic = True             # Decide si corres la simulación isotrópicamente o no

npe       = [1] * n                # e- primarios

pressure  = [1.001] * n               # bar

gap       = [0.7, 0.25, 0.15]                # mm

gas1      = ["C2H2F4"] * n   

mixture1  = [100.] * n            # % gas

gas2      = ["cf4"] * n

mixture2  = [0.]*n                 # % gas

fieldE    = [52724, 78010, 98677]   # V/cm, is_isotropic)

height = [0.98] * n         # Decide a la altura que quieras lanzar la simulación (0 = Pegado al catodo, 1 = Pegado al Anodo)

time = [5.370, 1.240, 0.612]  # ns 

######################################################################


# ---------------------------
# COMPILACIÓN ÚNICA
# ---------------------------

subprocess.run(["rm", "-rf", "build/"])
os.makedirs("build", exist_ok=True)
subprocess.run(["cmake", ".."], cwd="build")
subprocess.run("make -j$Nproc", shell=True, cwd="build")

root_dir = "rootArchives"

if is_isotropic:
    root_dir = "rootArchives_isotropic"
    os.makedirs(root_dir, exist_ok=True)
else:
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
        rootFileName,
        str(fieldE[i]),
        f"{gap[i]:.2f}",
        str(pressure[i]),
        str(npe[i]),
        gas1[i],
        f"{mixture1[i]:.3f}",
        gas2[i],
        f"{mixture2[i]:.3f}",
        str(height[i]),
        f"{time[i]:.4f}"
        
    ]

    all_jobs.append(args)

# ---------------------------
# MULTIPROCESSING
# ---------------------------

num_cores = min(cpu_count(), n)
print(f"\nUsando {num_cores} núcleos para las simulaciones...\n")

with Pool(processes=num_cores) as pool:
    pool.map(partial(run_fatGemC, is_isotropic=is_isotropic), all_jobs)

print("\n✔ Todas las simulaciones han terminado.\n")
