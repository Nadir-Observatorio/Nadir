"""Command line of nadir.

python -m nadir banco comprobar --bruto F:\\Biblioteca\\bruto
python -m nadir banco registrar --carpeta <folder> --archivo <file> --periodo 2012-2024 --url <url>
python -m nadir cobertura comprobar --bruto F:\\Biblioteca\\bruto [--anios 2023 2024]
python -m nadir fuego comprobar --bruto F:\\Biblioteca\\bruto [--anios 2024]
python -m nadir fuego cicatrices --bruto F:\\Biblioteca\\bruto --salida scars.csv [--anios 2024]
python -m nadir celdas construir --bruto F:\\Biblioteca\\bruto --resolucion 7 --salida <folder>
python -m nadir elevacion descargar --bruto F:\\Biblioteca\\bruto
python -m nadir elevacion pendiente --bruto F:\\Biblioteca\\bruto --salida <folder>
"""

import argparse
import sys
from pathlib import Path

from nadir import banco, celdas, exportacion
from nadir.capas import cobertura, elevacion, fuego
from nadir.errors import NadirError


def _ha(valor: float) -> str:
    return f"{valor:,.1f} ha"


def _grilla(grilla: exportacion.Grilla) -> str:
    return (
        f"{grilla.ancho} x {grilla.alto} pixels, cell {grilla.paso!r} degrees, "
        f"top-left corner ({grilla.oeste!r}, {grilla.norte!r})"
    )


def _anios(pedidos: list[int], todos: tuple[int, ...]) -> tuple[int, ...]:
    if not pedidos:
        return todos
    fuera = sorted(set(pedidos) - set(todos))
    if fuera:
        raise NadirError(f"years outside {todos[0]}-{todos[-1]}: {fuera}")
    return tuple(sorted(set(pedidos)))


def _banco_comprobar(args: argparse.Namespace) -> None:
    raiz = Path(args.bruto)
    carpetas = sorted(p.parent for p in raiz.glob(f"mapbiomas-chile/*/{banco.PROCEDENCIA}"))
    if not carpetas:
        raise NadirError(f"no MapBiomas folders with {banco.PROCEDENCIA} under {raiz}")
    for carpeta in carpetas:
        archivos = banco.comprobar(carpeta)
        print(f"{carpeta}: {len(archivos)} files match their record")


def _banco_registrar(args: argparse.Namespace) -> None:
    archivo = banco.registrar(
        Path(args.carpeta), args.archivo, args.periodo, args.url, args.url_verificada, args.fecha
    )
    print(f"{archivo.nombre}: {archivo.bytes} bytes, sha256 {archivo.sha256}, recorded")


def _cobertura_comprobar(args: argparse.Namespace) -> None:
    anios = _anios(args.anios, cobertura.ANIOS_ESTUDIO)
    with cobertura.abrir(Path(args.bruto)) as mosaico:
        print(f"{cobertura.PREFIJO}: {len(mosaico.teselas)} tiles, {_grilla(mosaico.grilla)}")
        informe = cobertura.describir(mosaico, anios)
    for anio in anios:
        hectareas = informe.hectareas[anio]
        dentro = sum(h for c, h in hectareas.items() if c != cobertura.FUERA)
        print(f"{anio}: {_ha(dentro)} inside the export area")
        for codigo, valor in hectareas.items():
            if codigo == cobertura.FUERA:
                continue
            clase = cobertura.NIVEL_1.get(codigo, "no_data")
            print(f"  {codigo:>3} {clase:<30} {_ha(valor):>18}")


def _fuego_comprobar(args: argparse.Namespace) -> None:
    anios = _anios(args.anios, fuego.ANIOS)
    bruto = Path(args.bruto)
    anual, mensual = fuego.abrir(bruto)
    with anual, mensual:
        print(f"{fuego.PREFIJO_ANUAL}: {len(anual.teselas)} tiles, {_grilla(anual.grilla)}")
        print(f"{fuego.PREFIJO_MENSUAL}: {len(mensual.teselas)} tiles, same grid")
        informe = fuego.describir(anual, mensual, anios)
    for anio in anios:
        meses = ", ".join(f"{m}: {_ha(h)}" for m, h in informe.hectareas_mes[anio].items())
        print(f"{anio}: {_ha(informe.hectareas_anio[anio])} burned  ({meses or 'no fire'})")
    if (bruto / cobertura.CARPETA / banco.PROCEDENCIA).exists():
        try:
            with cobertura.abrir(bruto) as cob:
                columna, fila = exportacion.desfase(cob.grilla, informe.grilla)
        except NadirError as error:
            print(f"Land cover grid: not compared ({error})")
        else:
            print(f"Land cover grid: aligned, offset of {columna} columns and {fila} rows")


def _fuego_cicatrices(args: argparse.Namespace) -> None:
    anios = _anios(args.anios, fuego.ANIOS)
    salida = Path(args.salida)
    if salida.exists():
        raise NadirError(f"refusing to overwrite {salida}")
    anual, mensual = fuego.abrir(Path(args.bruto))
    todas: list[fuego.Cicatriz] = []
    with anual, mensual:
        for anio in anios:
            lista = fuego.cicatrices(anual, anio)
            todas.extend(lista)
            if not lista:
                print(f"{anio}: no scars")
                continue
            mayor = max(lista, key=lambda c: c.hectareas)
            print(
                f"{anio}: {len(lista)} scars, {_ha(sum(c.hectareas for c in lista))}; "
                f"largest {_ha(mayor.hectareas)}, longest box diagonal "
                f"{max(c.diagonal_km for c in lista):,.1f} km"
            )
    fuego.escribir_cicatrices(salida, todas)
    print(f"{salida}: {len(todas)} scars written")


def _celdas_construir(args: argparse.Namespace) -> None:
    with cobertura.abrir(Path(args.bruto)) as mosaico:
        banda = cobertura.banda(cobertura.ANIOS_ESTUDIO[-1])
        mapa = celdas.construir(mosaico, banda, args.resolucion, Path(args.salida))
    print(
        f"{mapa.carpeta}: {mapa.celdas.size} cells at resolution {mapa.resolucion}, "
        f"area of the land cover export"
    )


def _elevacion_descargar(args: argparse.Namespace) -> None:
    bruto = Path(args.bruto)
    nombres = elevacion.necesarias(bruto)
    print(f"{len(nombres)} tiles cover the land cover export")
    nuevos, ausentes = elevacion.descargar(bruto, nombres)
    for archivo in nuevos:
        print(f"{archivo.nombre}: {archivo.bytes} bytes, sha256 {archivo.sha256}")
    for tesela in ausentes:
        print(f"{tesela}: not in the archive (ocean only)")
    total = len(banco.comprobar(bruto / elevacion.CARPETA))
    print(f"{len(nuevos)} downloaded now; {total} tiles in the bank")


def _elevacion_pendiente(args: argparse.Namespace) -> None:
    bruto = Path(args.bruto)
    with cobertura.abrir(bruto) as mosaico:
        banda = cobertura.banda(cobertura.ANIOS_ESTUDIO[-1])
        registro = elevacion.construir_pendiente(bruto, mosaico, banda, Path(args.salida))
        g = mosaico.grilla
    tramos = registro["pixeles_por_tramo"]
    if not isinstance(tramos, list):
        raise NadirError("slope record without pixel counts")
    total = sum(tramos)
    nombres = ("below 10", "10 to below 25", "25 or more")
    for nombre, cuenta in zip(nombres, tramos, strict=True):
        print(f"slope {nombre} degrees: {cuenta} pixels ({100 * cuenta / max(total, 1):.1f} %)")
    print(f"pixels inside the export without elevation: {registro['pixeles_dentro_sin_dem']}")
    print(f"{args.salida}: {g.ancho} x {g.alto} pixels written")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m nadir")
    temas = parser.add_subparsers(dest="tema", required=True)

    ban = temas.add_parser("banco", help="raw-data bank")
    ban_acc = ban.add_subparsers(dest="accion", required=True)
    ban_comp = ban_acc.add_parser("comprobar", help="check every recorded file")
    ban_comp.add_argument("--bruto", required=True, help="root of the raw-data bank")
    ban_comp.set_defaults(funcion=_banco_comprobar)
    ban_reg = ban_acc.add_parser("registrar", help="record a file already in a bank folder")
    ban_reg.add_argument("--carpeta", required=True, help="bank folder with its procedencia.toml")
    ban_reg.add_argument("--archivo", required=True, help="file name inside the folder")
    ban_reg.add_argument("--periodo", required=True, help="period the file covers")
    ban_reg.add_argument("--url", required=True, help="where it came from")
    ban_reg.add_argument(
        "--url-verificada", action="store_true", help="the URL downloads exactly this file"
    )
    ban_reg.add_argument("--fecha", default=None, help="arrival date; default: file time")
    ban_reg.set_defaults(funcion=_banco_registrar)

    cob = temas.add_parser("cobertura", help="MapBiomas Chile land cover, collection 2.0")
    cob_acc = cob.add_subparsers(dest="accion", required=True)
    cob_comp = cob_acc.add_parser("comprobar", help="check the export and report area by code")
    cob_comp.add_argument("--bruto", required=True, help="root of the raw-data bank")
    cob_comp.add_argument("--anios", type=int, nargs="+", default=[], help="years (default: all)")
    cob_comp.set_defaults(funcion=_cobertura_comprobar)

    fue = temas.add_parser("fuego", help="MapBiomas Fire Chile, collection 1")
    fue_acc = fue.add_subparsers(dest="accion", required=True)
    fue_comp = fue_acc.add_parser("comprobar", help="check both exports and report burned area")
    fue_comp.add_argument("--bruto", required=True, help="root of the raw-data bank")
    fue_comp.add_argument("--anios", type=int, nargs="+", default=[], help="years (default: all)")
    fue_comp.set_defaults(funcion=_fuego_comprobar)
    fue_cic = fue_acc.add_parser("cicatrices", help="fire scars by year (8 neighbours), to CSV")
    fue_cic.add_argument("--bruto", required=True, help="root of the raw-data bank")
    fue_cic.add_argument("--salida", required=True, help="CSV to write; never overwritten")
    fue_cic.add_argument("--anios", type=int, nargs="+", default=[], help="years (default: all)")
    fue_cic.set_defaults(funcion=_fuego_cicatrices)

    cel = temas.add_parser("celdas", help="pixel-to-cell maps of the MapBiomas exports")
    cel_acc = cel.add_subparsers(dest="accion", required=True)
    cel_con = cel_acc.add_parser("construir", help="build the map once; never overwritten")
    cel_con.add_argument("--bruto", required=True, help="root of the raw-data bank")
    cel_con.add_argument("--resolucion", type=int, required=True, help="H3 resolution, 5 to 8")
    cel_con.add_argument("--salida", required=True, help="folder for the map")
    cel_con.set_defaults(funcion=_celdas_construir)

    ele = temas.add_parser("elevacion", help="Copernicus DEM GLO-30 tiles")
    ele_acc = ele.add_subparsers(dest="accion", required=True)
    ele_des = ele_acc.add_parser("descargar", help="download the tiles the study area needs")
    ele_des.add_argument("--bruto", required=True, help="root of the raw-data bank")
    ele_des.set_defaults(funcion=_elevacion_descargar)
    ele_pen = ele_acc.add_parser(
        "pendiente", help="slope bands and altitude on the land cover grid"
    )
    ele_pen.add_argument("--bruto", required=True, help="root of the raw-data bank")
    ele_pen.add_argument("--salida", required=True, help="folder for the files; never overwritten")
    ele_pen.set_defaults(funcion=_elevacion_pendiente)

    args = parser.parse_args(argv)
    try:
        args.funcion(args)
    except NadirError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
