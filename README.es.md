*Read in English: [README.md](README.md)*

# Nadir: observatorio de riesgo y exposición

Observatorio abierto de peligro y vulnerabilidad socioambiental de Chile. Nadir une variables
territoriales de muchos proveedores en una grilla de celdas H3, con la fuente, la licencia y el
método de cada cifra a la vista.

**Estado:** solo el motor. Este repositorio contiene el sustrato común (celdas H3), el registro de
variables, las reglas de licencias, los registros de procedencia, el motor de capas y los lectores
de las exportaciones de Earth Engine de MapBiomas Chile, probados con datos sintéticos.
Todavía no publica ninguna cifra.

## Contenido

```text
registro/variables.toml   variables, con su licencia y la atribución que pide cada proveedor
src/nadir/                errors, licencias, registro, procedencia, grilla, tabla, nucleo, banco,
                          area, exportacion, celdas, transiciones
src/nadir/capas/          un módulo por fuente (cobertura: MapBiomas Chile, colección 2.0;
                          fuego: MapBiomas Fuego Chile, colección 1; elevacion: Copernicus GLO-30)
gee/                      scripts de exportación de Google Earth Engine
tests/                    un módulo de pruebas por módulo, solo con datos sintéticos
docs/design.md            principios, sustrato, licencias, motor de capas y reglas estadísticas
```

## Uso

Python 3.12.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux y macOS: source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q
ruff check . && mypy
```

Los datos originales van a un banco fuera del repositorio, una carpeta por fuente con su
`procedencia.toml` (origen, fecha, tamaño y sha256 de cada archivo; las entradas solo se agregan).
Los mapas de MapBiomas Chile se exportan desde Google Earth Engine con
`gee/exportar_mapbiomas_7_regiones.js`, y el zip de Google Drive se registra tal como llega:

```bash
python -m nadir banco registrar --carpeta <banco>/mapbiomas-chile/fuego-c1 --archivo <zip> --periodo 2013-2025 --url earthengine:<activo>
python -m nadir banco comprobar --bruto <banco>                        # cada archivo contra su registro
python -m nadir fuego comprobar --bruto <banco> --anios 2024           # área quemada por año y mes
python -m nadir cobertura comprobar --bruto <banco> --anios 2023 2024  # área por código de la leyenda
python -m nadir fuego cicatrices --bruto <banco> --salida <archivo.csv>   # cicatrices por año (8 vecinos)
python -m nadir celdas construir --bruto <banco> --resolucion 7 --salida <carpeta>  # mapa de píxeles a celdas, una vez
python -m nadir elevacion descargar --bruto <banco>                        # teselas Copernicus GLO-30 del área de estudio
python -m nadir elevacion pendiente --bruto <banco> --salida <carpeta>     # tramos de pendiente y altitud en la grilla de Cobertura
python -m nadir lfs construir --bruto <banco> --salida <carpeta>           # Landscape Fire Scars en la grilla de Cobertura
python -m nadir transiciones tabular --bruto <banco> --celdas <carpeta> --desfases=-1,2,3 --anios 2013 2021 --salida <archivo.parquet>
```

`transiciones tabular` acepta además `--resolucion`, `--pendiente <carpeta>` (filas separadas por
tramo de pendiente) y `--cicatrices <carpeta>` con `--excluir` o `--excluir-control`
(`ignicion_2008_2012`, `hasta_2018`) para dejar fuera los píxeles dentro de cicatrices conocidas.

El código y sus mensajes están en inglés; los nombres de los módulos, en español.

## Licencia

Código: [MIT](LICENSE). Cada capa de datos lleva su propia licencia, registrada en
`registro/variables.toml` y en el `procedencia.json` de cada capa (ver `docs/design.md`,
sección 4).

## Autor

Hecho por [Camilo López Barra](https://github.com/camilopezbarra), geógrafo.
