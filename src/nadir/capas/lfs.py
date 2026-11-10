"""Landscape Fire Scars Database for Chile (Miranda et al. 2022), read inside its zip files.

The dataset ships one zip per region and period. Each fire comes as seven files; this module uses
two of them: the binary scar raster `FireScar_<region>_ID<id>_u<threshold>_<YYYYMMDD>.tif`
(1 = scar) and the attribute table of the scar polygons (`.dbf` with the same name), which holds
the scar's total area, season and overlapping scars. The date in the name is the ignition date
recorded by CONAF. Nothing is extracted: members are read from the zips in the bank.

Scars are carried onto an export grid by centre: each export pixel takes the value of the scar
pixel that holds its centre (half-open cells: a centre on a shared edge goes to the cell east or
south of it). Two products come out of one pass over the zips:

- a layer of flags per export pixel (`BIT_HASTA_2018`: inside any scar of the dataset, seasons
  1985 to 2018; `BIT_IGNICION_2008_2012`: inside a scar whose CONAF ignition date is between
  2008-01-01 and 2012-12-31);
- an inventory of every scar (identifier, region, season, ignition, threshold, total area,
  overlaps, zip member and its sha256, pixels on the grid), and the export pixels of the scars
  chosen for the comparison with MapBiomas Fire (seasons and minimum area given by the caller).
"""

import hashlib
import json
import re
import struct
import zipfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
from rasterio.io import MemoryFile

from nadir import banco, exportacion
from nadir.errors import SourceError

CARPETA = Path("lfs-chile") / "pangaea-941127"
PAGINA = "https://doi.org/10.1594/PANGAEA.941127"
BIT_HASTA_2018 = 1
BIT_IGNICION_2008_2012 = 2
IGNICION_DESDE = date(2008, 1, 1)
IGNICION_HASTA = date(2012, 12, 31)

_NOMBRE = re.compile(
    r"^FireScar_(?P<region>CL-[A-Z]+)_ID(?P<id>\d+)_u(?P<umbral>\d+)_(?P<fecha>\d{8})\.tif$"
)
_BANDERAS = "banderas.u8"
_INVENTARIO = "inventario.parquet"
_PIXELES = "pixeles_comparacion.parquet"
_REGISTRO = "lfs.json"
_TOLERANCIA = 1e-9


@dataclass(frozen=True)
class Cicatriz:
    """One scar of the dataset, as named in its zip and described in its attribute table."""

    id: int
    region: str
    umbral: int
    ignicion: date
    temporada: int
    area_total_m2: float
    solapes: str
    zip: str
    miembro: str


def temporada(ignicion: date) -> int:
    """Season of an ignition date: July of the year before to June of the year named."""
    return ignicion.year + 1 if ignicion.month >= 7 else ignicion.year


def banderas_de(ignicion: date) -> int:
    """Flags that a scar with this CONAF ignition date puts on its pixels."""
    bits = BIT_HASTA_2018
    if IGNICION_DESDE <= ignicion <= IGNICION_HASTA:
        bits |= BIT_IGNICION_2008_2012
    return bits


def leer_dbf(datos: bytes) -> list[dict[str, str]]:
    """Records of a dBASE III table, as text; deleted records are skipped."""
    if len(datos) < 32:
        raise SourceError("attribute table shorter than its header")
    registros, cabecera, largo = struct.unpack("<IHH", datos[4:12])
    campos: list[tuple[str, int]] = []
    posicion = 32
    while posicion < cabecera - 1 and datos[posicion] != 0x0D:
        nombre = datos[posicion : posicion + 11].split(b"\x00", 1)[0].decode("ascii")
        campos.append((nombre, datos[posicion + 16]))
        posicion += 32
    if not campos or cabecera + registros * largo > len(datos):
        raise SourceError("attribute table without fields or shorter than its records")
    filas: list[dict[str, str]] = []
    for i in range(registros):
        inicio = cabecera + i * largo
        if datos[inicio : inicio + 1] == b"*":
            continue
        fila: dict[str, str] = {}
        cursor = inicio + 1
        for nombre, ancho in campos:
            crudo = datos[cursor : cursor + ancho]
            fila[nombre] = crudo.decode("latin-1").strip()
            cursor += ancho
        filas.append(fila)
    return filas


def _miembros(archivo: zipfile.ZipFile) -> Iterator[tuple[str, re.Match[str]]]:
    for miembro in archivo.namelist():
        coincide = _NOMBRE.match(miembro.rsplit("/", 1)[-1])
        if coincide:
            yield miembro, coincide


def _cicatriz(
    archivo: zipfile.ZipFile, zip_nombre: str, miembro: str, m: re.Match[str]
) -> Cicatriz:
    texto = m.group("fecha")
    try:
        ignicion = date(int(texto[:4]), int(texto[4:6]), int(texto[6:]))
    except ValueError:
        raise SourceError(f"{miembro}: ignition date {texto} is not a date") from None
    dbf = miembro[: -len(".tif")] + ".dbf"
    try:
        filas = leer_dbf(archivo.read(dbf))
    except KeyError:
        raise SourceError(f"{zip_nombre}: {miembro} has no attribute table {dbf}") from None
    if not filas:
        raise SourceError(f"{zip_nombre}: {dbf} has no records")
    primera = filas[0]
    try:
        area = float(primera["TotalArea"])
        declarada = int(float(primera["Season"]))
    except (KeyError, ValueError):
        raise SourceError(f"{zip_nombre}: {dbf} lacks a numeric TotalArea or Season") from None
    calculada = temporada(ignicion)
    if declarada != calculada:
        raise SourceError(
            f"{miembro}: season {declarada} in the table, {calculada} from the ignition date"
        )
    return Cicatriz(
        id=int(m.group("id")),
        region=m.group("region"),
        umbral=int(m.group("umbral")),
        ignicion=ignicion,
        temporada=declarada,
        area_total_m2=area,
        solapes=primera.get("OverlapIDs", ""),
        zip=zip_nombre,
        miembro=miembro,
    )


def proyectar(
    contenido: bytes, grilla: exportacion.Grilla
) -> tuple[int, int, npt.NDArray[np.bool_]]:
    """Export pixels inside a scar raster: top row, left column and a boolean window.

    Each export pixel takes the scar pixel that holds its centre; pixels whose centre falls
    outside the scar raster, or on its no-data, are outside the scar.
    """
    with MemoryFile(contenido) as memoria, memoria.open() as fuente:
        if fuente.crs is None or fuente.crs.to_epsg() != 4326:
            raise SourceError(f"scar raster in {fuente.crs}, expected EPSG:4326")
        t = fuente.transform
        if abs(t.b) > _TOLERANCIA or abs(t.d) > _TOLERANCIA or t.a <= 0 or t.e >= 0:
            raise SourceError("scar raster is not north-up")
        valores = fuente.read(1)
        sin_dato = fuente.nodata
    oeste, norte, paso_x, paso_y = t.c, t.f, t.a, -t.e
    este = oeste + paso_x * valores.shape[1]
    sur = norte - paso_y * valores.shape[0]
    p = grilla.paso
    f0 = max(0, int(np.ceil((grilla.norte - norte) / p - 0.5)))
    f1 = min(grilla.alto - 1, int(np.floor((grilla.norte - sur) / p - 0.5)))
    c0 = max(0, int(np.ceil((oeste - grilla.oeste) / p - 0.5)))
    c1 = min(grilla.ancho - 1, int(np.floor((este - grilla.oeste) / p - 0.5)))
    if f1 < f0 or c1 < c0:
        return 0, 0, np.zeros((0, 0), dtype=bool)
    latitudes = grilla.latitudes(f0, f1 - f0 + 1)
    longitudes = grilla.longitudes(c0, c1 - c0 + 1)
    filas = np.floor((norte - latitudes) / paso_y).astype(np.int64)
    columnas = np.floor((longitudes - oeste) / paso_x).astype(np.int64)
    ok_f = (filas >= 0) & (filas < valores.shape[0])
    ok_c = (columnas >= 0) & (columnas < valores.shape[1])
    ventana = np.zeros((filas.size, columnas.size), dtype=bool)
    f_ok = np.clip(filas, 0, valores.shape[0] - 1)
    c_ok = np.clip(columnas, 0, valores.shape[1] - 1)
    sub = valores[f_ok][:, c_ok]
    marca = sub == 1
    if sin_dato is not None:
        marca &= sub != sin_dato
    ventana[np.ix_(ok_f, ok_c)] = marca[np.ix_(ok_f, ok_c)]
    return f0, c0, ventana


def _sha256(ruta: Path) -> str:
    resumen = hashlib.sha256()
    with ruta.open("rb") as origen:
        while bloque := origen.read(1 << 20):
            resumen.update(bloque)
    return resumen.hexdigest()


def construir(
    bruto: Path,
    grilla: exportacion.Grilla,
    carpeta: Path,
    temporadas_comparacion: tuple[int, ...] = (2014, 2015, 2016, 2017, 2018),
    area_minima_ha: float = 250.0,
) -> dict[str, object]:
    """Flags, inventory and comparison pixels of every scar in the bank, saved in `carpeta`.

    The scars kept for the comparison are those of `temporadas_comparacion` with a total area
    above `area_minima_ha` (strictly greater) that touch the grid. Never overwrites.
    """
    if (carpeta / _REGISTRO).exists() or (carpeta / _BANDERAS).exists():
        raise SourceError(f"refusing to overwrite the scar files in {carpeta}")
    origen = bruto / CARPETA
    toml = origen / banco.PROCEDENCIA
    zips = _zips_registrados(bruto)
    banderas = np.zeros((grilla.alto, grilla.ancho), dtype=np.uint8)
    inventario: dict[str, list[object]] = {k: [] for k in (
        "id", "region", "umbral", "ignicion", "temporada", "area_total_m2", "solapes",
        "zip", "miembro", "sha256_miembro", "pixeles_en_grilla", "comparacion",
    )}
    comp_id: list[npt.NDArray[np.int64]] = []
    comp_fila: list[npt.NDArray[np.uint32]] = []
    comp_col: list[npt.NDArray[np.uint32]] = []
    vistos: set[int] = set()
    for ruta in zips:
        with zipfile.ZipFile(ruta) as archivo:
            for miembro, m in _miembros(archivo):
                cic = _cicatriz(archivo, ruta.name, miembro, m)
                if cic.id in vistos:
                    raise SourceError(f"scar ID{cic.id} appears twice ({ruta.name})")
                vistos.add(cic.id)
                contenido = archivo.read(miembro)
                f0, c0, ventana = proyectar(contenido, grilla)
                pixeles = int(ventana.sum())
                bits = banderas_de(cic.ignicion)
                if pixeles:
                    vista = banderas[f0 : f0 + ventana.shape[0], c0 : c0 + ventana.shape[1]]
                    vista[ventana] |= np.uint8(bits)
                elegida = (
                    pixeles > 0
                    and cic.temporada in temporadas_comparacion
                    and cic.area_total_m2 > area_minima_ha * 10_000
                )
                if elegida:
                    ff, cc = np.nonzero(ventana)
                    comp_id.append(np.full(ff.size, cic.id, dtype=np.int64))
                    comp_fila.append((ff + f0).astype(np.uint32))
                    comp_col.append((cc + c0).astype(np.uint32))
                fila = {**asdict(cic), "ignicion": cic.ignicion.isoformat()}
                for clave, valor in fila.items():
                    inventario[clave].append(valor)
                inventario["sha256_miembro"].append(hashlib.sha256(contenido).hexdigest())
                inventario["pixeles_en_grilla"].append(pixeles)
                inventario["comparacion"].append(elegida)
    if not vistos:
        raise SourceError(f"no FireScar rasters found in the zips of {origen}")
    carpeta.mkdir(parents=True, exist_ok=True)
    parcial = carpeta / f"{_BANDERAS}.part"
    banderas.tofile(parcial)
    del banderas
    parcial.replace(carpeta / _BANDERAS)
    tabla = pa.table(inventario).sort_by([("id", "ascending")])
    pq.write_table(tabla, carpeta / _INVENTARIO)
    vacio = np.zeros(0, dtype=np.int64)
    pixeles_tabla = pa.table({
        "id": pa.array(np.concatenate(comp_id) if comp_id else vacio, type=pa.int64()),
        "fila": pa.array(np.concatenate(comp_fila) if comp_fila else vacio.astype(np.uint32)),
        "columna": pa.array(np.concatenate(comp_col) if comp_col else vacio.astype(np.uint32)),
    })
    pq.write_table(pixeles_tabla, carpeta / _PIXELES)
    elegidas = sum(1 for e in inventario["comparacion"] if e)
    en_grilla = sum(1 for p in inventario["pixeles_en_grilla"] if isinstance(p, int) and p > 0)
    registro: dict[str, object] = {
        "grilla": asdict(grilla),
        "regla": "each export pixel takes the scar pixel holding its centre",
        "bits": {
            "hasta_2018": BIT_HASTA_2018,
            "ignicion_2008_2012": BIT_IGNICION_2008_2012,
        },
        "ignicion_2008_2012": [IGNICION_DESDE.isoformat(), IGNICION_HASTA.isoformat()],
        "comparacion": {
            "temporadas": list(temporadas_comparacion),
            "area_minima_ha_mayor_que": area_minima_ha,
            "cicatrices": elegidas,
        },
        "cicatrices": len(vistos),
        "cicatrices_en_grilla": en_grilla,
        "procedencia_toml_sha256": _sha256(toml),
        "zips": {r.name: r.stat().st_size for r in zips},
        "sha256": {n: _sha256(carpeta / n) for n in (_BANDERAS, _INVENTARIO, _PIXELES)},
    }
    texto = json.dumps(registro, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (carpeta / _REGISTRO).write_bytes(texto.encode("utf-8"))
    return registro


def leer_banderas(
    carpeta: Path, grilla: exportacion.Grilla, fila: int, filas: int
) -> npt.NDArray[np.uint8]:
    """Scar flags of rows `fila` to `fila + filas - 1`, after checking the saved grid."""
    try:
        registro = json.loads((carpeta / _REGISTRO).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SourceError(f"no scar layer in {carpeta}") from None
    if registro.get("grilla") != asdict(grilla):
        raise SourceError(f"the scar layer in {carpeta} belongs to another grid")
    with (carpeta / _BANDERAS).open("rb") as origen:
        origen.seek(fila * grilla.ancho)
        datos = np.fromfile(origen, dtype="u1", count=filas * grilla.ancho)
    if datos.size != filas * grilla.ancho:
        raise SourceError(f"{carpeta / _BANDERAS} is shorter than its grid")
    return datos.reshape(filas, grilla.ancho)


def _zips_registrados(bruto: Path) -> list[Path]:
    """The zip files of the bank folder, each checked by name and size against its record."""
    origen = bruto / CARPETA
    zips = sorted(origen.glob("*.zip"))
    if not zips:
        raise SourceError(f"no zip files in {origen}")
    toml = origen / banco.PROCEDENCIA
    if not toml.is_file():
        raise SourceError(f"{origen} has no {banco.PROCEDENCIA}")
    _, archivos = banco.leer(origen)
    registrados = {a.nombre: a for a in archivos}
    for ruta in zips:
        registro_zip = registrados.get(ruta.name)
        if registro_zip is None:
            raise SourceError(f"{ruta.name} is not recorded in {toml}")
        if ruta.stat().st_size != registro_zip.bytes:
            raise SourceError(f"{ruta.name}: size differs from its record")
    return zips


def recorrer(
    bruto: Path, grilla: exportacion.Grilla
) -> Iterator[tuple[Cicatriz, int, int, npt.NDArray[np.bool_]]]:
    """Every scar of the bank with its footprint on the grid: top row, left column and window.

    Same reading and centre rule as `construir`; scars that miss the grid have an empty window.
    """
    for ruta in _zips_registrados(bruto):
        with zipfile.ZipFile(ruta) as archivo:
            for miembro, m in _miembros(archivo):
                cic = _cicatriz(archivo, ruta.name, miembro, m)
                f0, c0, ventana = proyectar(archivo.read(miembro), grilla)
                yield cic, f0, c0, ventana


_CONTEO = "conteo.u8"
_TEMPORADA_UNICA = "temporada_unica.u8"
_REGISTRO_CONTEO = "conteo.json"
CONTEO_TOPE = 254
TEMPORADA_BASE = 2000


def construir_conteo(
    bruto: Path, grilla: exportacion.Grilla, carpeta: Path, hasta: int = 2018
) -> dict[str, object]:
    """Number of scars of the seasons up to `hasta` on each pixel, and the season of the scar
    where there is exactly one.

    Two files, one byte per pixel on the grid: `conteo.u8` (number of scars, capped at 254) and
    `temporada_unica.u8` (season minus 2000 where the count is one, else 0). Same reading and
    centre rule as `construir`; a scar repeated across zips counts once. Never overwrites.
    """
    for nombre in (_CONTEO, _TEMPORADA_UNICA, _REGISTRO_CONTEO):
        if (carpeta / nombre).exists():
            raise SourceError(f"refusing to overwrite {carpeta / nombre}")
    conteo = np.zeros((grilla.alto, grilla.ancho), dtype=np.uint8)
    temporada = np.zeros((grilla.alto, grilla.ancho), dtype=np.uint8)
    vistas = en_grilla = 0
    ids: set[int] = set()
    for cic, f0, c0, ventana in recorrer(bruto, grilla):
        if cic.temporada > hasta or cic.id in ids:
            continue
        ids.add(cic.id)
        vistas += 1
        if not ventana.any():
            continue
        en_grilla += 1
        alto, ancho = ventana.shape
        zona_c = conteo[f0 : f0 + alto, c0 : c0 + ancho]
        zona_t = temporada[f0 : f0 + alto, c0 : c0 + ancho]
        zona_t[ventana & (zona_c == 0)] = cic.temporada - TEMPORADA_BASE
        zona_t[ventana & (zona_c >= 1)] = 0
        zona_c[ventana] = np.minimum(zona_c[ventana].astype(np.int16) + 1, CONTEO_TOPE)
    if vistas == 0:
        raise SourceError(f"no scar of the seasons up to {hasta} in the bank")
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / _CONTEO).write_bytes(conteo.tobytes())
    (carpeta / _TEMPORADA_UNICA).write_bytes(temporada.tobytes())
    registro: dict[str, object] = {
        "grilla": asdict(grilla),
        "hasta_temporada": hasta,
        "cicatrices": vistas,
        "cicatrices_en_grilla": en_grilla,
        "pixeles_con_una": int((conteo == 1).sum()),
        "pixeles_con_dos_o_mas": int((conteo >= 2).sum()),
        "sha256": {n: _sha256(carpeta / n) for n in (_CONTEO, _TEMPORADA_UNICA)},
    }
    texto = json.dumps(registro, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (carpeta / _REGISTRO_CONTEO).write_bytes(texto.encode("utf-8"))
    return registro


def leer_conteo(
    carpeta: Path, grilla: exportacion.Grilla, fila: int, filas: int
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Scar count and single-scar season (minus 2000) of rows `fila` to `fila + filas - 1`."""
    try:
        registro = json.loads((carpeta / _REGISTRO_CONTEO).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SourceError(f"no scar count layer in {carpeta}") from None
    if not exportacion.misma_grilla(exportacion.Grilla(**registro["grilla"]), grilla):
        raise SourceError(f"the scar count layer in {carpeta} belongs to another grid")
    salida = []
    for nombre in (_CONTEO, _TEMPORADA_UNICA):
        with (carpeta / nombre).open("rb") as origen:
            origen.seek(fila * grilla.ancho)
            datos = np.fromfile(origen, dtype="u1", count=filas * grilla.ancho)
        if datos.size != filas * grilla.ancho:
            raise SourceError(f"{carpeta / nombre} is shorter than its grid")
        salida.append(datos.reshape(filas, grilla.ancho))
    return salida[0], salida[1]
