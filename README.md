*Leer en español: [README.es.md](README.es.md)*

# Nadir: risk and exposure observatory

Open observatory of socio-environmental hazard and vulnerability in Chile. Nadir joins territorial
variables from many providers on one grid of H3 cells, with the source, licence and method of every
figure in view.

**Status:** engine only. This repository holds the common substrate (H3 cells), the variable
registry, the licence rules, provenance records, the layer engine and the readers of the Earth
Engine exports of MapBiomas Chile, all tested on synthetic data. It publishes no figure yet.

## Contents

```text
registro/variables.toml   variables, with licence and attribution as each provider asks for them
src/nadir/                errors, licencias, registro, procedencia, grilla, tabla, nucleo, banco,
                          area, exportacion, celdas, transiciones
src/nadir/capas/          one module per source (cobertura: MapBiomas Chile land cover, collection 2.0;
                          fuego: MapBiomas Fire Chile, collection 1; elevacion: Copernicus GLO-30)
gee/                      Google Earth Engine export scripts
tests/                    one test module per source module, synthetic data only
docs/design.md            principles, substrate, licences, layer engine and statistical rules
```

## Use

Python 3.12.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux and macOS: source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q
ruff check . && mypy
```

Raw data go to a bank folder outside the repository, one folder per source with its own
`procedencia.toml` (origin, date, size and sha256 of every file; entries are only appended).
MapBiomas Chile maps are exported from Google Earth Engine with `gee/exportar_mapbiomas_7_regiones.js`
and the zip from Google Drive is recorded as received:

```bash
python -m nadir banco registrar --carpeta <bank>/mapbiomas-chile/fuego-c1 --archivo <zip> --periodo 2013-2025 --url earthengine:<asset>
python -m nadir banco comprobar --bruto <bank>                        # every file against its record
python -m nadir fuego comprobar --bruto <bank> --anios 2024           # burned area by year and month
python -m nadir cobertura comprobar --bruto <bank> --anios 2023 2024  # area by legend code
python -m nadir fuego cicatrices --bruto <bank> --salida <file.csv>    # fire scars by year (8 neighbours)
python -m nadir celdas construir --bruto <bank> --resolucion 7 --salida <folder>  # pixel-to-cell map, once
python -m nadir elevacion descargar --bruto <bank>                       # Copernicus GLO-30 tiles of the study area
python -m nadir elevacion pendiente --bruto <bank> --salida <folder>     # slope bands and altitude on the land cover grid
```

## Licence

Code: [MIT](LICENSE). Each data layer carries its own licence, recorded in
`registro/variables.toml` and in the `procedencia.json` of each layer (see `docs/design.md`,
section 4).

## Author

Made by [Camilo López Barra](https://github.com/camilopezbarra), geographer.
