from pathlib import Path

import pytest

from nadir.errors import RegistryError
from nadir.registro import CAMPOS, cargar

REGISTRO = Path(__file__).resolve().parent.parent / "registro" / "variables.toml"

BASE = {
    "id": "proveedor.producto.anual",
    "proveedor": "Proveedor",
    "producto": "Producto",
    "version": "1",
    "licencia": "CC-BY-4.0",
    "atribucion": "Texto de atribución",
    "unidad": "fracción",
    "tiempo": [2000, 2010],
    "agregacion": "fraccion",
    "sin_dato": "excluido",
    "url": "https://ejemplo.org/",
}


def _toml(entradas: list[dict[str, object]]) -> str:
    partes = []
    for entrada in entradas:
        partes.append("[[variable]]")
        for clave, valor in entrada.items():
            if isinstance(valor, list):
                partes.append(f"{clave} = [{', '.join(str(v) for v in valor)}]")
            else:
                partes.append(f'{clave} = "{valor}"')
    return "\n".join(partes) + "\n"


def _escribir(tmp_path: Path, entradas: list[dict[str, object]]) -> Path:
    ruta = tmp_path / "variables.toml"
    ruta.write_text(_toml(entradas), encoding="utf-8")
    return ruta


def test_project_registry_is_valid():
    variables = cargar(REGISTRO)
    assert "mapbiomas.fuego.c1.anual" in variables
    assert variables["mapbiomas.fuego.c1.mensual"].agregacion == "moda"
    assert variables["mapbiomas.fuego.c1.frecuencia"].tiempo == (2025, 2025)
    assert variables["mapbiomas.cobertura.c2.clase"].tiempo == (1999, 2024)


def test_valid_entry_loads(tmp_path):
    variables = cargar(_escribir(tmp_path, [BASE]))
    assert list(variables) == ["proveedor.producto.anual"]
    assert variables["proveedor.producto.anual"].tiempo == (2000, 2010)


@pytest.mark.parametrize("campo", CAMPOS)
def test_missing_field_is_refused(tmp_path, campo):
    entrada = {k: v for k, v in BASE.items() if k != campo}
    with pytest.raises(RegistryError, match="missing fields"):
        cargar(_escribir(tmp_path, [entrada]))


def test_unknown_field_is_refused(tmp_path):
    with pytest.raises(RegistryError, match="unknown fields"):
        cargar(_escribir(tmp_path, [{**BASE, "color": "rojo"}]))


def test_unknown_licence_is_refused(tmp_path):
    from nadir.errors import LicenseError

    with pytest.raises(LicenseError):
        cargar(_escribir(tmp_path, [{**BASE, "licencia": "propietaria"}]))


def test_unknown_rule_is_refused(tmp_path):
    with pytest.raises(RegistryError, match="aggregation"):
        cargar(_escribir(tmp_path, [{**BASE, "agregacion": "mediana"}]))


def test_bad_id_is_refused(tmp_path):
    with pytest.raises(RegistryError, match="id must look like"):
        cargar(_escribir(tmp_path, [{**BASE, "id": "Sin Puntos"}]))


def test_reversed_period_is_refused(tmp_path):
    with pytest.raises(RegistryError, match="starts after"):
        cargar(_escribir(tmp_path, [{**BASE, "tiempo": [2010, 2000]}]))


def test_plain_http_is_refused(tmp_path):
    with pytest.raises(RegistryError, match="https"):
        cargar(_escribir(tmp_path, [{**BASE, "url": "http://ejemplo.org/"}]))


def test_duplicated_id_is_refused(tmp_path):
    with pytest.raises(RegistryError, match="duplicated"):
        cargar(_escribir(tmp_path, [BASE, BASE]))


def test_missing_registry_is_reported(tmp_path):
    with pytest.raises(RegistryError, match="not found"):
        cargar(tmp_path / "no.toml")


def test_invalid_toml_is_reported(tmp_path):
    ruta = tmp_path / "variables.toml"
    ruta.write_text("[[variable]\n", encoding="utf-8")
    with pytest.raises(RegistryError, match="not valid TOML"):
        cargar(ruta)
