# Simulación de campo eléctrico uniforme

Este módulo simula avalanchas electrónicas microscópicas en un campo eléctrico uniforme con Garfield++ y Magboltz. Mantiene los tres modos actuales de trabajo:

- `mode = 0`: gap y campo fijos.
- `mode = 1`: ganancia fijada y cálculo del gap mediante el ajuste de `alpha`.
- `mode = 2`: ganancia fijada y cálculo del campo mediante el ajuste de `alpha`.

La configuración habitual se realiza en `runUniform_multithread.py`.

## Estructura

```text
.
├── CMakeLists.txt
├── uniformE.cxx
├── runUniform_multithread.py
├── importing.py
├── gainCalculation.py
├── gas_data.csv
├── fitPlots/
├── rootArchives/
└── rootBackup/
```

Cuando se activa la creación del GIF, se utiliza además:

```text
space_charge_gif/
```

## Dependencias

- Garfield++
- ROOT
- GSL
- CMake
- Python 3
- NumPy, pandas, uproot, matplotlib, SciPy y tqdm

## Ejecución

```bash
python3 runUniform_multithread.py
```

El script recompila el ejecutable, lanza los trabajos en paralelo, actualiza los ROOT de `rootBackup/`, reconstruye `gas_data.csv` y genera los ajustes de `alpha` usados por los modos 1 y 2.

## Nuevos controles microscópicos

Se configuran al principio de `runUniform_multithread.py`:

```python
enable_space_charge = False
make_gif = False
max_electron_energy_inputs = 200_000
```

- `enable_space_charge`: activa o desactiva la acumulación de los iones positivos como anillos cargados. Los iones producidos por una avalancha afectan a las avalanchas primarias posteriores. No se realiza propagación iónica.
- `make_gif`: activa o desactiva el GIF de la primera avalancha de cada trabajo.
- `max_electron_energy_inputs`: máximo de energías almacenadas para construir `hElectronEnergyDistribution`. El límite duro es `200000`.

Las excitaciones no se guardan evento a evento. Se acumulan durante la simulación en histogramas de tamaño fijo, por lo que el peso del ROOT no crece con el número de excitaciones.

Los doce primeros argumentos del ejecutable C++ no han cambiado. Los controles anteriores se pasan como argumentos opcionales al final de la línea de comandos.

## Contenido de los ROOT

Los nombres comunes se han uniformado con el repositorio de avalanchas:

- `gasData`
- `dataPerPrimaryElectron`
- `dataPerElectron`

También se guardan:

- `hElectronEnergyDistribution`: distribución energética obtenida durante el transporte microscópico, incluyendo los pasos del algoritmo de null collisions y limitada al máximo configurado mediante muestreo de reservorio.
- `hLevels`: número de excitaciones electrónicas por nivel de Magboltz.
- `hExcXY`: distribución transversal conjunta de las posiciones de excitación.
- `hExcZT`: distribución conjunta de profundidad y tiempo de las excitaciones. El eje temporal se amplía automáticamente si aparecen tiempos fuera del rango inicial, manteniendo fijo el número de bins.

`DataExc` se ha eliminado. Para reconstruir excitaciones en los proyectos consumidores se muestrea por Monte Carlo:

1. `(x, y)` desde `hExcXY`.
2. `(z, t)` desde `hExcZT`.
3. el nivel desde `hLevels`.

Esta aproximación conserva las correlaciones `x-y` y `z-t`, pero asume independencia entre la estructura transversal, la evolución longitudinal-temporal y el nivel excitado.

`dataPerPrimaryElectron` contiene únicamente:

- `ne`: electrones finales de la avalancha primaria.
- `ni`: iones producidos.
- `npe`: número de electrones primarios representados por la entrada; vale `1` en cada fila.

## Cálculo de alpha

Cada ROOT nuevo guarda en `gasData`:

- `npe`
- `neTotal`, `niTotal`
- `neMean`, `niMean`
- `gainSim = <ne>`
- `alphaEff = ln(gainSim) / gap_cm`
- `alphaFromNe`, `alphaFromNi`
- presión, gap, campo y composición del gas
- estado de space charge, GIF y estadísticas de los nuevos outputs

`importing.py` usa `gasData` por defecto, pero conserva lectura de `dataOfGas` para poder reutilizar ROOT antiguos.
