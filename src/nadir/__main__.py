"""Command line of nadir.

python -m nadir banco comprobar --bruto F:\\Biblioteca\\bruto
python -m nadir banco registrar --carpeta <folder> --archivo <file> --periodo 2012-2024 --url <url>
python -m nadir cobertura comprobar --bruto F:\\Biblioteca\\bruto [--anios 2023 2024]
python -m nadir cobertura-c1 preparar --bruto F:\\Biblioteca\\bruto
python -m nadir cobertura-c1 comprobar --bruto F:\\Biblioteca\\bruto [--anios 2012 2022]
python -m nadir fuego comprobar --bruto F:\\Biblioteca\\bruto [--anios 2024]
python -m nadir fuego cicatrices --bruto F:\\Biblioteca\\bruto --salida scars.csv [--anios 2024]
python -m nadir celdas construir --bruto F:\\Biblioteca\\bruto --resolucion 7 --salida <folder>
python -m nadir elevacion descargar --bruto F:\\Biblioteca\\bruto
python -m nadir elevacion pendiente --bruto F:\\Biblioteca\\bruto --salida <folder>
python -m nadir interfaz construir --bruto F:\\Biblioteca\\bruto --salida <folder> [--anios 2017]
python -m nadir lfs construir --bruto F:\\Biblioteca\\bruto --salida <folder>
python -m nadir lfs conteo --bruto F:\\Biblioteca\\bruto --salida <folder>
python -m nadir transiciones tabular --bruto F:\\Biblioteca\\bruto --celdas <folder>
    --desfases=-1,2,3 --anios 2013 2021 --salida <table.parquet>
    [--resolucion 7] [--pendiente <folder>] [--cicatrices <folder>]
    [--excluir ignicion_2008_2012] [--excluir-control hasta_2018]
python -m nadir estudio tablas --bruto F:\\Biblioteca\\bruto --celdas <folder> --pendiente <folder>
    --interfaz <folder> --salida <folder> [--tablas k0 k2 k3 k5 balance]
    [--cicatrices <folder>] [--excluir ignicion_2008_2012] [--excluir-control hasta_2018]
    [--conteo-lfs <folder> --tablas l5]
python -m nadir estudio tablas ... --coleccion 1 --salida <folder>   (k3 with collection 1)
python -m nadir estudio analizar --tablas <folder> --salida <results.json> [--replicas 2000]
python -m nadir estudio trayectorias --tablas <folder> --salida <results.json>
python -m nadir estudio urbanizacion --tablas <folder> [--tabla u5|l5] --salida <results.json>
python -m nadir estudio sensibilidad --tablas <folder> --numero 2 --salida <results.json>
python -m nadir estudio mapas --tablas <folder> --salida <folder>
python -m nadir estudio sensibilidad7 --tablas <folder> --tablas-c1 <folder> --salida <results.json>
python -m nadir estudio chequeo-lfs --bruto F:\\Biblioteca\\bruto --salida <folder>
python -m nadir estudio validar --bruto F:\\Biblioteca\\bruto --clases <clases.tif>
    --comunas <comunas.geojson> --emsr <EMSR715_products.zip> --salida <validation.json>
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import nadir
from nadir import banco, celdas, exportacion, transiciones
from nadir.capas import cobertura, cobertura_c1, elevacion, fuego, interfaz, lfs
from nadir.errors import NadirError
from nadir.estudio import analisis, exploratorio, mapas, tablas, trayectorias, validacion

_BITS = {"hasta_2018": lfs.BIT_HASTA_2018, "ignicion_2008_2012": lfs.BIT_IGNICION_2008_2012}


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


def _cobertura_c1_preparar(args: argparse.Namespace) -> None:
    carpeta = Path(args.bruto) / cobertura_c1.CARPETA
    banco.preparar(carpeta, cobertura_c1.FUENTE)
    print(f"{carpeta}: ready; copy the export zip here and record it with banco registrar")


def _cobertura_c1_comprobar(args: argparse.Namespace) -> None:
    anios = _anios(args.anios, cobertura_c1.ANIOS_EXPORTADOS)
    with cobertura_c1.abrir(Path(args.bruto)) as mosaico:
        print(f"{cobertura_c1.PREFIJO}: {len(mosaico.teselas)} tiles, {_grilla(mosaico.grilla)}")
        print("same grid as collection 2")
        informe = cobertura_c1.describir(mosaico, anios)
    for anio in anios:
        hectareas = informe.hectareas[anio]
        dentro = sum(h for c, h in hectareas.items() if c != cobertura.FUERA)
        print(f"{anio}: {_ha(dentro)} inside the export area")
        for codigo, valor in hectareas.items():
            if codigo == cobertura.FUERA:
                continue
            clase = cobertura_c1.NIVEL_1.get(codigo, "no_data")
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


def _interfaz_construir(args: argparse.Namespace) -> None:
    anios = _anios(args.anios, interfaz.ANIOS)
    with cobertura.abrir(Path(args.bruto)) as mosaico:
        registro = interfaz.construir(mosaico, Path(args.salida), anios)
    por_anio = registro["anios"]
    if not isinstance(por_anio, dict):
        raise NadirError("interface record without years")
    for anio, datos in por_anio.items():
        print(
            f"{anio}: {datos['pixeles_cerca']} of {datos['pixeles_dentro']} pixels within "
            f"{interfaz.CORTE_M:.0f} m; band sha256 {datos['sha256_banda']}"
        )
    print(f"{args.salida}: {len(por_anio)} years written")


def _estudio_tablas(args: argparse.Namespace) -> None:
    insumos = tablas.Insumos(
        Path(args.celdas),
        Path(args.pendiente),
        Path(args.interfaz),
        Path(args.cicatrices) if args.cicatrices else None,
        sum({_BITS[n] for n in args.excluir}),
        sum({_BITS[n] for n in args.excluir_control}),
        Path(args.conteo_lfs) if args.conteo_lfs else None,
    )
    elegidas = tuple(args.tablas) if args.coleccion == 2 else (tablas.TABLA_C1,)
    huellas = tablas.construir(
        Path(args.bruto), insumos, Path(args.salida), elegidas, args.coleccion
    )
    for nombre, huella in huellas.items():
        print(f"{nombre}: content hash {huella}")
    print(f"{args.salida}: {len(huellas)} tables written")


def _estudio_analizar(args: argparse.Namespace) -> None:
    salida = Path(args.salida)
    if salida.exists():
        raise NadirError(f"refusing to overwrite {salida}")
    resultados = analisis.analizar(Path(args.tablas), args.replicas)
    analisis.escribir(resultados, salida)
    h1 = resultados["H1"]
    if "ic95_pp" in h1:
        bajo, alto = (float(x) for x in h1["ic95_pp"])
        print(
            f"H1: effect {h1['efecto_pp']} pp, 95 % interval {bajo} to {alto}, "
            f"Holm p {h1['p_holm']} (m = 2)"
        )
    else:
        print(f"H1: effect {h1['efecto_pp']} pp, {h1['nota']}")
    print("H3: not evaluable by construction (thematic filter of MapBiomas Fire)")
    print(f"{salida}: written with {args.replicas} replicates")


def _estudio_trayectorias(args: argparse.Namespace) -> None:
    salida = Path(args.salida)
    if salida.exists():
        raise NadirError(f"refusing to overwrite {salida}")
    resultados = trayectorias.trayectorias(Path(args.tablas))
    analisis.escribir(resultados, salida)
    print(f"{salida}: trajectories at k = 2, 3 and 5 and the k = 3 matrix by fire year")


def _estudio_urbanizacion(args: argparse.Namespace) -> None:
    salida = Path(args.salida)
    if salida.exists():
        raise NadirError(f"refusing to overwrite {salida}")
    resultados = exploratorio.urbanizacion(Path(args.tablas), args.replicas, args.tabla)
    analisis.escribir(resultados, salida)
    print(f"{salida}: exploratory urbanisation after fire, {args.replicas} replicates")


def _estudio_sensibilidad(args: argparse.Namespace) -> None:
    salida = Path(args.salida)
    if salida.exists():
        raise NadirError(f"refusing to overwrite {salida}")
    resultados = analisis.sensibilidad_lfs(Path(args.tablas), args.numero, args.replicas)
    analisis.escribir(resultados, salida)
    for clave, valor in resultados.items():
        if isinstance(valor, dict) and "ic95_pp" in valor:
            bajo, alto = (float(x) for x in valor["ic95_pp"])
            print(f"{clave}: effect {valor['efecto_pp']} pp, 95 % interval {bajo} to {alto}")
    print(f"{salida}: written with {args.replicas} replicates")


def _estudio_sensibilidad7(args: argparse.Namespace) -> None:
    salida = Path(args.salida)
    if salida.exists():
        raise NadirError(f"refusing to overwrite {salida}")
    resultados = analisis.sensibilidad_coleccion(
        Path(args.tablas), Path(args.tablas_c1), args.replicas
    )
    analisis.escribir(resultados, salida)
    for clave, valor in resultados.items():
        if isinstance(valor, dict) and "efecto_pp" in valor:
            intervalo = valor.get("ic95_pp")
            texto = f", 95 % interval {intervalo[0]} to {intervalo[1]}" if intervalo else ""
            print(f"{clave}: effect {valor['efecto_pp']} pp{texto}")
    print(f"{salida}: written with {args.replicas} replicates")


def _estudio_mapas(args: argparse.Namespace) -> None:
    celdas = mapas.por_celda(Path(args.tablas))
    rutas = mapas.escribir(celdas, Path(args.salida))
    for res, t in celdas.items():
        con = int(t.efecto_pp.notna().sum())
        print(f"resolution {res}: {len(t)} cells with burned area, {con} with an effect")
    for ruta in rutas:
        print(f"{ruta}: written")


def _estudio_chequeo_lfs(args: argparse.Namespace) -> None:
    salida = Path(args.salida)
    if salida.exists():
        raise NadirError(f"refusing to overwrite {salida}")
    filas, resumen = validacion.chequeo_lfs(Path(args.bruto))
    if not filas:
        raise NadirError("no scar of the 2014-2018 seasons above 250 ha touches the grid")
    salida.mkdir(parents=True)
    with (salida / "jaccard_8_1.csv").open("w", encoding="utf-8", newline="") as destino:
        escritor = csv.DictWriter(destino, fieldnames=list(filas[0]), lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)
    texto = json.dumps(resumen, ensure_ascii=False, indent=1, allow_nan=False) + "\n"
    (salida / "resumen_8_1.json").write_bytes(texto.encode())
    print(
        f"{resumen['cicatrices']} scars: mean Jaccard {resumen['jaccard_media']}, "
        f"median {resumen['jaccard_mediana']}"
    )
    print(f"{salida}: written")


def _estudio_validar(args: argparse.Namespace) -> None:
    salida = Path(args.salida)
    if salida.exists():
        raise NadirError(f"refusing to overwrite {salida}")
    resultados = validacion.validar(
        Path(args.bruto), Path(args.clases), Path(args.comunas), Path(args.emsr)
    )
    texto = json.dumps(resultados, ensure_ascii=False, indent=1, allow_nan=False) + "\n"
    salida.write_bytes(texto.encode())
    exactitud = resultados["8.3"]
    bajo, alto = exactitud["exactitud_global_ic95"]
    print(
        f"accuracy against EMSR715: {exactitud['exactitud_global_pct']} % "
        f"({float(bajo)} to {float(alto)}); commission {exactitud['comision_pct']} %, "
        f"omission {exactitud['omision_pct']} %"
    )
    print(f"{salida}: written")


def _sha256(ruta: Path) -> str:
    resumen = hashlib.sha256()
    with ruta.open("rb") as origen:
        while bloque := origen.read(1 << 20):
            resumen.update(bloque)
    return resumen.hexdigest()


def _lfs_construir(args: argparse.Namespace) -> None:
    bruto = Path(args.bruto)
    with cobertura.abrir(bruto) as mosaico:
        grilla = mosaico.grilla
    registro = lfs.construir(bruto, grilla, Path(args.salida))
    comparacion = registro["comparacion"]
    if not isinstance(comparacion, dict):
        raise NadirError("scar record without its comparison block")
    print(
        f"{registro['cicatrices']} scars read; {registro['cicatrices_en_grilla']} touch the grid; "
        f"{comparacion['cicatrices']} kept for the comparison"
    )
    print(f"{args.salida}: flags, inventory and comparison pixels written")


def _lfs_conteo(args: argparse.Namespace) -> None:
    bruto = Path(args.bruto)
    with cobertura.abrir(bruto) as mosaico:
        grilla = mosaico.grilla
    registro = lfs.construir_conteo(bruto, grilla, Path(args.salida))
    print(
        f"{registro['cicatrices']} scars up to 2018; {registro['pixeles_con_una']} pixels with "
        f"one, {registro['pixeles_con_dos_o_mas']} with two or more"
    )
    print(f"{args.salida}: scar count and single-scar season written")


def _desfases(texto: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in texto.split(","))
    except ValueError:
        raise NadirError(f"offsets must be integers separated by commas, not {texto!r}") from None


def _transiciones_tabular(args: argparse.Namespace) -> None:
    bruto = Path(args.bruto)
    salida = Path(args.salida)
    if salida.exists():
        raise NadirError(f"refusing to overwrite {salida}")
    if len(args.anios) != 2 or args.anios[0] > args.anios[1]:
        raise NadirError("--anios takes the first and last fire year")
    anios = tuple(range(args.anios[0], args.anios[1] + 1))
    desfases = _desfases(args.desfases)
    excluir = sum(_BITS[n] for n in set(args.excluir))
    excluir_control = sum(_BITS[n] for n in set(args.excluir_control))
    pendiente = Path(args.pendiente) if args.pendiente else None
    cicatrices = Path(args.cicatrices) if args.cicatrices else None
    procedencia = {
        "nadir": nadir.__version__,
        "fuego_procedencia_sha256": _sha256(bruto / fuego.CARPETA / banco.PROCEDENCIA),
        "cobertura_procedencia_sha256": _sha256(bruto / cobertura.CARPETA / banco.PROCEDENCIA),
        "celdas_mapa_sha256": _sha256(Path(args.celdas) / "mapa.json"),
        "anios": f"{anios[0]}-{anios[-1]}",
    }
    if pendiente is not None:
        procedencia["pendiente_sha256"] = _sha256(pendiente / "pendiente.json")
    if cicatrices is not None:
        procedencia["cicatrices_sha256"] = _sha256(cicatrices / "lfs.json")
    with cobertura.abrir(bruto) as cob:
        anual, mensual = fuego.abrir(bruto)
        with anual, mensual:
            mapa = celdas.abrir(Path(args.celdas), cob.grilla)
            tabla = transiciones.tabular(
                mapa,
                cob,
                anual,
                anios,
                desfases,
                args.resolucion,
                pendiente=pendiente,
                cicatrices=cicatrices,
                excluir_bits=excluir,
                excluir_bits_control=excluir_control,
            )
    contenido = transiciones.escribir(salida, tabla, procedencia)
    print(f"{salida}: {len(tabla)} rows, fire years {anios[0]}-{anios[-1]}, offsets {desfases}")
    print(f"  pixels left out without slope band: {_ha(tabla.excluido_sin_pendiente_m2 / 1e4)}")
    print(f"  pixels left out by scar flags: {_ha(tabla.excluido_por_cicatrices_m2 / 1e4)}")
    print(f"  hash {contenido}")


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

    c1 = temas.add_parser("cobertura-c1", help="MapBiomas Chile land cover, collection 1.0")
    c1_acc = c1.add_subparsers(dest="accion", required=True)
    c1_pre = c1_acc.add_parser("preparar", help="create the bank folder with its source block")
    c1_pre.add_argument("--bruto", required=True, help="root of the raw-data bank")
    c1_pre.set_defaults(funcion=_cobertura_c1_preparar)
    c1_comp = c1_acc.add_parser("comprobar", help="check the export, its grid and area by code")
    c1_comp.add_argument("--bruto", required=True, help="root of the raw-data bank")
    c1_comp.add_argument("--anios", type=int, nargs="+", default=[], help="years (default: all)")
    c1_comp.set_defaults(funcion=_cobertura_c1_comprobar)

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

    itf = temas.add_parser("interfaz", help="distance to the urban interface (land cover code 24)")
    itf_acc = itf.add_subparsers(dest="accion", required=True)
    itf_con = itf_acc.add_parser("construir", help="band and distance by year on the export grid")
    itf_con.add_argument("--bruto", required=True, help="root of the raw-data bank")
    itf_con.add_argument("--salida", required=True, help="folder for the files; never overwritten")
    itf_con.add_argument("--anios", type=int, nargs="*", default=[], help="years (default all)")
    itf_con.set_defaults(funcion=_interfaz_construir)

    lf = temas.add_parser("lfs", help="Landscape Fire Scars Database for Chile")
    lf_acc = lf.add_subparsers(dest="accion", required=True)
    lf_con = lf_acc.add_parser("construir", help="scar flags and inventory on the export grid")
    lf_con.add_argument("--bruto", required=True, help="root of the raw-data bank")
    lf_con.add_argument("--salida", required=True, help="folder for the files; never overwritten")
    lf_con.set_defaults(funcion=_lfs_construir)
    lf_cnt = lf_acc.add_parser("conteo", help="scars per pixel and the season of a single scar")
    lf_cnt.add_argument("--bruto", required=True, help="root of the raw-data bank")
    lf_cnt.add_argument("--salida", required=True, help="folder for the files; never overwritten")
    lf_cnt.set_defaults(funcion=_lfs_conteo)

    tra = temas.add_parser("transiciones", help="land cover transitions by H3 cell")
    tra_acc = tra.add_subparsers(dest="accion", required=True)
    tra_tab = tra_acc.add_parser("tabular", help="area of each combination of classes, to Parquet")
    tra_tab.add_argument("--bruto", required=True, help="root of the raw-data bank")
    tra_tab.add_argument("--celdas", required=True, help="folder of the resolution-8 cell map")
    tra_tab.add_argument(
        "--desfases", required=True, help="offsets from the fire year, e.g. --desfases=-1,2,3"
    )
    tra_tab.add_argument(
        "--anios", type=int, nargs=2, required=True, help="first and last fire year"
    )
    tra_tab.add_argument("--resolucion", type=int, default=7, help="H3 resolution of the rows")
    tra_tab.add_argument("--pendiente", default=None, help="folder of the slope bands")
    tra_tab.add_argument("--cicatrices", default=None, help="folder of the scar layer")
    tra_tab.add_argument(
        "--excluir", nargs="+", choices=sorted(_BITS), default=[], help="flags out of every group"
    )
    tra_tab.add_argument(
        "--excluir-control", nargs="+", choices=sorted(_BITS), default=[],
        help="flags out of the control group",
    )
    tra_tab.add_argument("--salida", required=True, help="Parquet file; never overwritten")
    tra_tab.set_defaults(funcion=_transiciones_tabular)

    est = temas.add_parser("estudio", help="the fire and land cover study")
    est_acc = est.add_subparsers(dest="accion", required=True)
    est_tab = est_acc.add_parser("tablas", help="area tables by resolution-7 cell, to Parquet")
    est_tab.add_argument("--bruto", required=True, help="root of the raw-data bank")
    est_tab.add_argument("--celdas", required=True, help="folder of the resolution-8 cell map")
    est_tab.add_argument("--pendiente", required=True, help="folder of the slope bands")
    est_tab.add_argument("--interfaz", required=True, help="folder of the interface layer")
    est_tab.add_argument("--salida", required=True, help="folder for the tables; never overwritten")
    est_tab.add_argument(
        "--tablas",
        nargs="+",
        choices=[*tablas.TABLAS, *tablas.EXPLORATORIAS, *tablas.EXPLORATORIAS_LFS, tablas.BALANCE],
        default=[*tablas.TABLAS, tablas.BALANCE],
        help="tables to build (balance needs k3)",
    )
    est_tab.add_argument("--cicatrices", default=None, help="folder of the scar layer")
    est_tab.add_argument("--conteo-lfs", default=None, help="folder of the scar count (for l5)")
    est_tab.add_argument(
        "--excluir", nargs="+", choices=sorted(_BITS), default=[], help="flags out of every group"
    )
    est_tab.add_argument(
        "--excluir-control",
        nargs="+",
        choices=sorted(_BITS),
        default=[],
        help="flags out of the control only",
    )
    est_tab.add_argument(
        "--coleccion",
        type=int,
        choices=tablas.COLECCIONES,
        default=2,
        help="land cover collection; 1 builds k3 for sensitivity 7 only",
    )
    est_tab.set_defaults(funcion=_estudio_tablas)
    est_ana = est_acc.add_parser("analizar", help="main analysis of the tables, to JSON")
    est_ana.add_argument("--tablas", required=True, help="folder of the study tables")
    est_ana.add_argument("--salida", required=True, help="results file; never overwritten")
    est_ana.add_argument("--replicas", type=int, default=2000, help="bootstrap replicates")
    est_ana.set_defaults(funcion=_estudio_analizar)
    est_tra = est_acc.add_parser("trayectorias", help="trajectories and matrix by year, to JSON")
    est_tra.add_argument("--tablas", required=True, help="folder of the study tables")
    est_tra.add_argument("--salida", required=True, help="results file; never overwritten")
    est_tra.set_defaults(funcion=_estudio_trayectorias)
    est_urb = est_acc.add_parser("urbanizacion", help="exploratory urbanisation after fire")
    est_urb.add_argument("--tablas", required=True, help="folder with the u5 or l5 table")
    est_urb.add_argument(
        "--tabla", choices=["u5", "l5"], default="u5", help="fire from Fire or from LFS"
    )
    est_urb.add_argument("--salida", required=True, help="results file; never overwritten")
    est_urb.add_argument("--replicas", type=int, default=2000, help="bootstrap replicates")
    est_urb.set_defaults(funcion=_estudio_urbanizacion)
    est_sen = est_acc.add_parser("sensibilidad", help="scar sensitivities 2 and 3, to JSON")
    est_sen.add_argument("--tablas", required=True, help="folder of tables with the exclusion")
    est_sen.add_argument("--numero", required=True, choices=["2", "3"], help="sensitivity")
    est_sen.add_argument("--salida", required=True, help="results file; never overwritten")
    est_sen.add_argument("--replicas", type=int, default=2000, help="bootstrap replicates")
    est_sen.set_defaults(funcion=_estudio_sensibilidad)
    est_s7 = est_acc.add_parser(
        "sensibilidad7", help="H1 with land cover collection 1 against collection 2, to JSON"
    )
    est_s7.add_argument("--tablas", required=True, help="folder of the collection 2 tables")
    est_s7.add_argument("--tablas-c1", required=True, help="folder of the collection 1 k3 table")
    est_s7.add_argument("--salida", required=True, help="results file; never overwritten")
    est_s7.add_argument("--replicas", type=int, default=2000, help="bootstrap replicates")
    est_s7.set_defaults(funcion=_estudio_sensibilidad7)
    est_map = est_acc.add_parser(
        "mapas", help="burned area and H1 effect by H3 cell (resolutions 6 and 7), GeoJSON and CSV"
    )
    est_map.add_argument("--tablas", required=True, help="folder of the tables")
    est_map.add_argument("--salida", required=True, help="output folder; never overwritten")
    est_map.set_defaults(funcion=_estudio_mapas)
    est_lfs = est_acc.add_parser("chequeo-lfs", help="Jaccard of each large scar against Fire")
    est_lfs.add_argument("--bruto", required=True, help="root of the raw-data bank")
    est_lfs.add_argument("--salida", required=True, help="folder for the files; never overwritten")
    est_lfs.set_defaults(funcion=_estudio_chequeo_lfs)
    est_val = est_acc.add_parser("validar", help="agreement and accuracy against the 2024 fire")
    est_val.add_argument("--bruto", required=True, help="root of the raw-data bank")
    est_val.add_argument("--clases", required=True, help="severity map clases.tif")
    est_val.add_argument("--comunas", required=True, help="the four municipalities, GeoJSON")
    est_val.add_argument("--emsr", required=True, help="EMSR715_products.zip")
    est_val.add_argument("--salida", required=True, help="results file; never overwritten")
    est_val.set_defaults(funcion=_estudio_validar)

    args = parser.parse_args(argv)
    try:
        args.funcion(args)
    except NadirError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
