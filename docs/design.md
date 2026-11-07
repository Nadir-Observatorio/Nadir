# Nadir: design

Nadir is an open observatory of socio-environmental hazard and vulnerability in Chile. It joins
territorial variables from many providers on one common substrate, so any variable can be crossed
with any other, with its source, licence and method in view.

## 1. Principles

1. **Every published figure has a chain of custody.** Original with URL, size, sha256 and date;
   derived table with a content hash; published page checked against the data automatically.
2. **No layer without a verified licence.** Each variable states its licence, its attribution as
   the provider asks for it, and the licence its derivatives inherit.
3. **No composite index without its recipe.** Method, weights, inputs and version are published;
   weights are value choices and are called so.
4. **A scale that does not stigmatise.** Social vulnerability is published by cell and by comuna,
   never at a finer scale without a separate decision.
5. **A record is not proof.** An administrative record is shown with its exact name and scope;
   Nadir does not claim contamination, blame or cause that the source does not claim.
6. **Dynamic in the browser, no server.** The site reads static files.

## 2. Common substrate: H3 cells

| Resolution | Mean area | Use |
|---|---|---|
| 5 | about 250 km² | National view |
| 6 | about 36 km² | Regional view |
| 7 | about 5 km² | Main analysis |
| 8 | about 0.7 km² | Case zones |

- Pixels go to the resolution-8 cell that contains their centre, weighted by their area in km²;
  coarser cells add up their children (section 5).
- `fraccion_valida` is the area with valid data in the cell divided by the cell area.
- Comunas appear as tables; their boundaries are used only to assign cells and are neither drawn
  nor redistributed.

## 3. Variable registry and cell tables

- `registro/variables.toml` lists every variable: provider, product, version, licence,
  attribution, unit, period, aggregation rule (`fraccion`, `suma`, `media`, `moda`, `conteo`),
  no-data handling and URL.
- One Parquet table per variable, resolution and year, with the columns `celda`, `tiempo`,
  `valor` and `fraccion_valida`. Its content hash is computed from the values, so it does not
  depend on the Parquet library, compression or operating system.

## 4. Licences

- A product carries the most restrictive licence among its inputs.
- MapBiomas Chile states its terms in two ways (CC BY in the platform notes; "non-commercial use
  or public interest purposes" in its FAQ). Until clarified in writing, its data are treated as
  possibly non-commercial, and combining them with a share-alike licence (for example CC BY-SA
  4.0 census data) is refused.

## 5. Layer engine

A layer turns one registry variable into cell tables: `descargar` fetches the originals,
`derivar` builds the tables, `construir` writes tables and `procedencia.json` without ever
overwriting, and `verificar` checks originals and tables on disk against that record.

Sources served only through Google Earth Engine (MapBiomas Chile) are exported to Google Drive
by a script kept in `gee/`, in the native grid of each asset, and the zip from Drive goes into the
bank as received, recorded with `banco registrar`. Nadir reads the tiles in place, without
unzipping, and before using a pixel checks that they form one image (reference system, cell size,
band names, data type, no-data value, each tile where its name says, no gap, no overlap). Each
export script is kept byte for byte as it ran, Drive folder name included, so its sha256 matches
the one recorded in the bank.

On a grid in degrees a pixel has no fixed area (about 754 m² at 33° S for MapBiomas). Every area
comes from the exact area of each row on the WGS84 ellipsoid (`area.py`), never from a pixel count
times a nominal size. Distances (for example the box diagonal of a fire scar) use the mean
sphere, within 0.5 % of the ellipsoid.

Assigning pixels to cells is done once per export and saved as a cell map (`celdas.py`); every
band, year and product is then tabulated from it as the area of each value in each cell. H3
children do not tile their parent exactly: on the MapBiomas grid about 7 % of pixels have a
resolution-7 centre cell that is not the parent of their resolution-8 cell. Nadir builds one
resolution-8 map, assigned by centre, and reaches coarser resolutions by adding up child cells,
so tables nest exactly across scales (blocks for resampling included); each table records its
resolution.

Land cover transitions (`transiciones.py`) add up, by cell, the area of each combination of
classes read at chosen offsets from a fire year, for pixels burned once, burned more than once,
and never burned. Classes keep their original legend codes, with the level-1 class beside them.
With slope bands, each row also carries the band of its pixels, so control strata can be
formed from the table; pixels without a band are left out and their area is reported.

Slope is computed on the native grid of the elevation model (Horn, with east-west and
north-south spacing from the ellipsoid at each latitude), and each land cover pixel takes the
slope and altitude of the elevation pixel that holds its centre, without averaging.

## 6. Statistical rules

1. Hypotheses are written and dated before the data are looked at; exploration is labelled.
2. Families of tests are corrected (Holm for confirmatory hypotheses, Benjamini-Hochberg at 5 % for
   broad exploration), and the number of tests is published.
3. Neighbouring cells are not independent: intervals come from a spatial block bootstrap, and
   residual Moran's I is reported.
4. Main results are repeated at resolutions 7 and 6 (scale effect).
5. What holds for a cell does not hold for each person or property in it.
6. Effect sizes with intervals come first; p-values are secondary.
7. Changes in classified products are compared against a baseline of change in control areas.
