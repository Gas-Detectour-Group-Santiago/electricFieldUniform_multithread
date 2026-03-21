#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 13:32:27 2025

@author: pablo
"""
import subprocess
import os
from multiprocessing import Pool, cpu_count

def run_fatGemC(args,is_isotropic):
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

n = 3

######################################################################
# Parametros de la simulación


is_isotropic = True             # Decide si corres la simulación isotrópicamente o no

npe       = [1]*3                 # e- primarios

pressure  = [1, 1, 1]               # bar

gap       = [0.57]*3               # mm

gas1      = ["cf4","ar","he"]   

mixture1  = [100.0,80.0,80.0]           # % gas

gas2      = ["ar","cf4","cf4"]

mixture2  = [0.0,20.0,20.0]*n                 # % gas

fieldE    = [10000, 10000, 10000]   # V/cm

height = [0.99, 0.98, 0.95]         # Decide a la altura que quieras lanzar la simulación (0 = Pegado al catodo, 1 = Pegado al Anodo)

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
        f"{mixture1[i]:.1f}",
        gas2[i],
        f"{mixture2[i]:.1f}",
    ]

    all_jobs.append(args)

# ---------------------------
# MULTIPROCESSING
# ---------------------------

num_cores = min(cpu_count(), n)
print(f"\nUsando {num_cores} núcleos para las simulaciones...\n")

with Pool(processes=num_cores) as pool:
    pool.map(run_fatGemC, all_jobs, is_isotropic)

print("\n✔ Todas las simulaciones han terminado.\n")
