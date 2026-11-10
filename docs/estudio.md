# Fire and land cover: study method

What land cover follows a fire in central Chile, 2013 to 2024? This document describes what the
code in `src/nadir/estudio/` computes. The method was fixed and dated on 2026-11-07, before
MapBiomas Fire and MapBiomas land cover were crossed; every later change is a numbered amendment,
listed at the end with its date and whether results had been seen.

## 1. Data

- MapBiomas Fire Chile, collection 1: annual and monthly burned area, 2013 to 2025.
- MapBiomas Chile land cover, collection 2.0, final classification, 2012 to 2024.
- Copernicus DEM GLO-30: slope (Horn) and altitude.
- Landscape Fire Scars Database for Chile (Miranda et al. 2022): sensitivity analyses and a
  reproduction check. It shares authors with MapBiomas Fire, so it is not an independent
  reference.
- Copernicus EMS EMSR715 and the severity map of the 2024 Viña del Mar fire (repository
  incendio-vina-2024): accuracy and agreement.

The study area is the seven regions of the Fire export (Valparaíso to La Araucanía), on the native
grid of the exports (EPSG:4326, about 754 m² per pixel at 33° S).

## 2. Groups, strata and tables

- **Burned:** pixels burned in exactly one year of 2013-2025. **Reburned:** burned in more than one
  year; described apart, labelled exploratory. **Control:** never burned in 2013-2025.
- A pixel enters a table when it has a valid land cover code (not 0, 27 or 255) in every year the
  table reads and a slope band (below 10°, 10° to below 25°, 25° or more).
- **Tables** (`estudio tablas`): area by resolution-7 H3 cell of each combination of classes in
  t - 1, t + k - 1 and t + k, where t is the fire year (k = 2, 3 and 5), with the slope band and
  the interface band of t - 1. The resolution-7 area is the sum of its resolution-8 children.
- **Interface** (`interfaz construir`): Euclidean distance to land cover code 24
  (infrastructure) on the native grid, in strips of 0.1° with WGS84 metric spacing; exact up to
  10 km. Near means less than 2 km.
- **Stratum:** a resolution-6 cell with its first ring, crossed with the slope band and the fire
  year. A stratum enters when its control holds at least 100 pixels.

## 3. Estimates

- Every proportion is weighted by area. The standardised control proportion weights each
  stratum's control proportion by the burned area of that stratum.
- **Block bootstrap:** blocks are resolution-4 cells; all blocks of the domain are resampled with
  replacement, with all their content; 2,000 replicates; seed 20240202; 95 % percentile
  intervals; one-sided p = share of replicates at or below zero.
- **Holm** over H1 and H3 (m = 2).
- **Moran's I** of the H1 residual by resolution-7 cell, first-ring neighbours, 999 permutations.
- **Balance:** standardised mean differences of altitude and of distance to the interface
  (capped at 10 km) between burned and standardised control.
- **Classification error:** e*, the differential error that would cancel an effect (the lower
  end of its interval), set against the user's error of the destination classes.

## 4. Hypotheses

- **H1.** Among pixels that were forest formation in t - 1, burning raises the persistent
  conversion to another level-1 class in t + 2 and t + 3, against the standardised control.
- **H3.** Among burned pixels with vegetation in t - 1, persistent conversion to non-vegetated area
  is higher within 2 km of the urban interface. **Not evaluable by construction** (amendment E-3):
  the thematic filter of MapBiomas Fire masks burned pixels with the land cover product itself, so
  no burned pixel ever holds non-vegetated area or water, and H3 compares zero with zero. The same
  filter likely biases H1 towards zero.

## 5. Sensitivity analyses

1. Without the 2016 and 2017 fire years.
2. Without the pixels of Landscape Fire Scars ignited from 2008-01-01 to 2012-12-31, in every
   group (amendment E-1).
3. Controls outside every Landscape Fire Scar of the 1985-2018 seasons.
4. Blocks at resolution 5; resolution 3 descriptive only (fewer than 30 blocks).
5. Strata at resolution 7 instead of 6.
6. Differential classification error (e*).

7. Land cover collection 1 instead of collection 2, fire years 2013-2019 against collection 2 in
   the same years (exploratory, amendments E-5 and E-7). Collection 1 must be on the grid of
   collection 2; nothing is resampled. Its codes take their level-1 class from the MapBiomas
   network legend, the same table collection 2 uses, extended with the network codes collection 2
   lacks (`nadir.capas.cobertura_c1`); any other code stops the reading. Each collection keeps its
   own valid pixels and reports the area it leaves out; the difference of the two effects is
   descriptive.

Effects by fire year are reported as description.

## 5b. Exploratory: urbanisation after fire

Declared before computing (amendment E-5): the share of burned land that is infrastructure
(code 24) in t + 4 and t + 5 against the standardised control, outside the window the Fire
filter is documented to mask (`estudio tablas --tablas u5`, then `estudio urbanizacion`).
**Not evaluable by construction:** no burned pixel holds code 24 in any year read, while the
control does. In the data the Fire filter removes every burned pixel that was ever
infrastructure (and ever water, code 33), not only over the four-year window its document
describes; codes 15 and 18 are reduced but present. The command reports this instead of an
effect.

The same question with the fire taken from Landscape Fire Scars, which does not go through the
MapBiomas filter (amendment E-6, declared before computing): seasons s = 2014-2018 (July of s - 1
to June of s); previous class in s - 2; outcome code 24 in s + 4 and s + 5; burned = inside exactly
one scar of 1985-2018, of season s; control = no scar of 1985-2018 and never burned in Fire
2013-2025; interface band of s - 2. Commands: `lfs conteo`, then `estudio tablas --tablas l5
--conteo-lfs <folder>` and `estudio urbanizacion --tabla l5`. The scar database reconstructs about
two thirds of the CONAF fires of 10 ha or more, so unrecorded burned land can sit in the control
and pull the effect towards zero.

## 6. Validation

- **Check against Landscape Fire Scars** (`estudio chequeo-lfs`): Jaccard index of each scar of
  the 2014-2018 seasons above 250 ha in the seven regions against the Fire burned area of the same
  season (July to June, from the monthly product), in the scar's box widened by 1 km, with other
  scars of the same season left out.
- **Agreement with the 2024 severity map** and **accuracy against EMSR715**
  (`estudio validar`): see the module docstring of `estudio/validacion.py`.

## 7. Reproduction

From the raw-data bank (`<banco>`), in this order:

```bash
python -m nadir celdas construir --bruto <banco> --resolucion 8 --salida <celdas>
python -m nadir elevacion pendiente --bruto <banco> --salida <pendiente>
python -m nadir interfaz construir --bruto <banco> --salida <interfaz>
python -m nadir lfs construir --bruto <banco> --salida <lfs>
python -m nadir estudio tablas --bruto <banco> --celdas <celdas> --pendiente <pendiente> --interfaz <interfaz> --salida <tablas> --tablas k0
python -m nadir estudio tablas ... --salida <tablas> --tablas k2
python -m nadir estudio tablas ... --salida <tablas> --tablas k3 balance
python -m nadir estudio tablas ... --salida <tablas> --tablas k5
python -m nadir estudio analizar --tablas <tablas> --salida resultados.json
python -m nadir estudio trayectorias --tablas <tablas> --salida trayectorias.json
python -m nadir estudio tablas ... --tablas k3 --cicatrices <lfs> --excluir ignicion_2008_2012 --salida <s2>
python -m nadir estudio sensibilidad --tablas <s2> --numero 2 --salida sensibilidad_2.json
python -m nadir estudio tablas ... --tablas k3 --cicatrices <lfs> --excluir-control hasta_2018 --salida <s3>
python -m nadir estudio sensibilidad --tablas <s3> --numero 3 --salida sensibilidad_3.json
python -m nadir cobertura-c1 comprobar --bruto <banco>
python -m nadir estudio tablas ... --coleccion 1 --salida <c1>
python -m nadir estudio sensibilidad7 --tablas <tablas> --tablas-c1 <c1> --salida sensibilidad_7.json
python -m nadir estudio chequeo-lfs --bruto <banco> --salida <chequeo>
pip install -e ".[validacion]"
python -m nadir estudio validar --bruto <banco> --clases <clases.tif> --comunas <comunas.geojson> --emsr <EMSR715_products.zip> --salida validacion.json
```

One table per process keeps memory below 8 GB. Every table carries a content hash in its
metadata, with the area left out for lack of a slope band, for an invalid land cover code and,
when used, for scars; the same inputs give the same bits.

## 8. Readings and implementation details

Where the method left room for a reading, the code takes this one:

- **Sensitivity 5** (resolution 6 against 7) changes the stratum of H1 to a resolution-7 cell
  with its first ring; blocks stay at resolution 4.
- **e\*** is set against the user's error of the destination classes weighted by the share of
  each destination in the changed burned area. That error is a share of the mapped area of the
  class, not of the burned area, so the comparison is conservative: it overstates the threat.
- **Moran's I** is computed for forest formation only, on unweighted residuals by resolution-7
  cell, with first-ring neighbours and a one-sided upper p from 999 permutations.
- **Reburned pixels** enter once per burn year; a later burn can fall inside the horizon.
- **Distance for the balance** is stored in steps of 40 m (rounding of 20 m at most). Using the
  column spacing of the strip's central latitude gives an error of up to about 1.4 m at 2 km at
  38° S. The area within 2 km of the edge of the export, where distance can be overestimated, is
  in the record of the interface layer.
- **Tables** are kept at resolution 7, as sums of their resolution-8 children; nothing changes in
  the figures.
- **Validation against the 2024 fire** reuses the sampling and parameters of the severity study of
  that fire (repository incendio-vina-2024), copied with their origin noted: 300 points per
  predicted stratum, a minimum Chebyshev distance of 10 pixels applied within each stratum,
  seed 20240202. Before the crossing, the monthly Fire export gave about 7,052 ha burned in
  February 2024 in the Valparaíso region against about 7,032 ha in the platform clip (0.3 %).
- **p values** of 0 are reported with their bound (below 1/2000 with 2,000 replicates).
- A class with fewer than 30 resolution-4 blocks with burned area is reported without interval.

## 9. Amendments

| # | Date | What it fixes | Results seen? |
|---|---|---|---|
| E-1 | 2026-11-08 | Sensitivity 2 removes scar pixels from every group, not only from the burned | No |
| E-2 | 2026-11-08 | The balance measures distance to the interface exactly up to 10 km, capped there; the 2 km cut of H3 does not change | No |
| E-3 | 2026-11-08 | The thematic filter of MapBiomas Fire is documented; H3 is reported as not evaluable by construction; Holm keeps m = 2; H1 is noted as biased towards zero; the omission of this filter when the method was fixed is declared | Yes |
| E-4 | 2026-11-08 | Implementation brought in line with the fixed method after an independent review of the code: H3 labelled not evaluable in every output and kept out of the adjustment of H1 (p of H1 doubled); balance computed on the same strata as the effect (control of 100 pixels or more); descriptive proportions reported with the burned area they rest on; minimum of 30 blocks checked for every interval; sensitivities 2 and 3 refuse tables with the wrong exclusion; left-out areas reported; the Landscape Fire Scars check removes neighbouring scars of every region and adds the median without overlaps | Yes |
| E-5 | 2026-11-09 | Two exploratory extensions declared before computing: robustness to land cover collection 1 (sensitivity 7) and urbanisation after fire; the second turned out not evaluable by construction | Partly (confirmatory results yes; the new figures no) |
| E-6 | 2026-11-09 | Urbanisation after fire with the fire defined by Landscape Fire Scars and the outcome from land cover collection 2 (exploratory) | Partly (confirmatory results yes; the new figures no) |
| E-7 | 2026-11-09 | Sensitivity 7 takes the level-1 class of each collection 1 code from the MapBiomas network legend shared by both collections, since the collection 1 legend document is no longer published; any code outside that table stops the reading | No collection 1 pixel had been read |
