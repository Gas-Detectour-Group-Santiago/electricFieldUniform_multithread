# electricUniform

Simulación microscópica de avalanchas en campo uniforme y búsqueda automática de los campos necesarios para alcanzar ganancias concretas.

## Uso normal

Edita `campaign.yaml` y ejecuta:

```bash
python3 run_campaign.py campaign.yaml
```

No hay que introducir campos eléctricos ni `npe`.

El programa:

1. lee los ROOT existentes;
2. calcula `alphaEffective = log(gain) / gap`;
3. ajusta, para cada mezcla, concentración y gap,

   ```text
   alphaEffective / p = A (E / p)^m exp[-(B / (E / p))^n]
   ```

4. propone automáticamente nuevos campos;
5. conserva todas las simulaciones, aunque no alcancen la ganancia buscada;
6. repite hasta quedar dentro de `gain_tolerance`;
7. adapta `npe` al coste de la avalancha y a su error estadístico;
8. ejecuta familias independientes en paralelo.

## Outputs

```text
outputs/
├── roots/
│   └── <mixture>/
│       └── gap_<gap>mm/
│           └── *.root
├── alpha/
│   └── <mixture>/
│       └── gap_<gap>mm.json
└── gifs/
    └── *.gif
```

`outputs/roots` contiene únicamente ROOTs. `outputs/alpha` contiene únicamente los puntos y parámetros necesarios para reconstruir e invertir las curvas de `alphaEffective`.

## Geometría

- Ánodo: `z = 0`.
- Lanzamiento del electrón: `z = gap`.
- Distancia usada en la ganancia y en `alphaEffective`: exactamente `gap`.
- Límite superior: `zMax = heightFactor × gap`.
- Valor por defecto: `heightFactor = 1.5`.
- Rango transversal de simulación, `hExcXY` y GIF: `x,y ∈ [-2 gap,+2 gap]`.

El factor de altura solo añade margen por encima del plano de lanzamiento. No cambia la distancia física empleada para calcular la ganancia.

Puede añadirse opcionalmente al YAML:

```yaml
height_factor: 1.5
```

## Contenido de cada ROOT

```text
gasData
dataPerPrimaryElectron: ne, ni, npe
dataPerElectron: status
hElectronEnergyDistribution
hLevels
hExcXY
hExcZT
```

Las excitaciones se reconstruyen por Monte Carlo muestreando de forma independiente:

- `(x, y)` desde `hExcXY`;
- `(z, t)` desde `hExcZT`;
- nivel excitado desde `hLevels`.

## GUI

```bash
python3 gui.py
```

La pestaña **Campaign** edita y ejecuta el YAML. La pestaña **GIF** permite seleccionar mezcla, concentración, presión, gap, campo o ganancia, `heightFactor`, `tMax`, número de frames, `npe`, space charge y movimiento de iones.

En el GIF los iones pueden mantenerse en su posición de creación o moverse hacia el plano superior con una velocidad constante configurable. La velocidad es un parámetro visual del GIF y no se usa en la simulación microscópica ni en el cálculo de `alphaEffective`. Cuando `space charge` está activo, las posiciones instantáneas de electrones e iones se emplean para reconstruir `Delta V_SC` en cada frame.

Los GIFs se generan aparte y no modifican los ajustes de alpha.

## Dependencias Python

```bash
python3 -m pip install -r requirements.txt
```

Además se necesitan ROOT, Garfield++, Magboltz y CMake.
