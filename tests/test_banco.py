import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from nadir import banco
from nadir.errors import ProvenanceError, SourceError

FUENTE = banco.Fuente(
    organismo="Proveedor",
    conjunto="Conjunto de prueba",
    fuente_pagina="https://ejemplo.org/",
    licencia="CC-BY-4.0",
    atribucion='Texto con "comillas" y \\ barra',
)

CONTENIDO = bytes(range(256)) * 5000


class Servidor(BaseHTTPRequestHandler):
    fallos_restantes = 0
    cortar = False
    pedidos = 0

    def log_message(self, *args):
        pass

    def _encabezados(self, largo):
        self.send_response(200)
        self.send_header("Content-Length", str(largo))
        self.end_headers()

    def do_HEAD(self):
        if self.path == "/no-existe.tif":
            self.send_error(404)
            return
        self._encabezados(len(CONTENIDO))

    def do_GET(self):
        Servidor.pedidos += 1
        if self.path == "/no-existe.tif":
            self.send_error(404)
            return
        if Servidor.fallos_restantes > 0:
            Servidor.fallos_restantes -= 1
            self.send_error(503)
            return
        self._encabezados(len(CONTENIDO))
        self.wfile.write(CONTENIDO[: len(CONTENIDO) // 2] if Servidor.cortar else CONTENIDO)


@pytest.fixture
def servidor(monkeypatch):
    monkeypatch.setattr(banco, "_https", lambda url: None)
    Servidor.fallos_restantes = 0
    Servidor.cortar = False
    Servidor.pedidos = 0
    http = ThreadingHTTPServer(("127.0.0.1", 0), Servidor)
    hilo = threading.Thread(target=http.serve_forever, daemon=True)
    hilo.start()
    yield f"http://127.0.0.1:{http.server_address[1]}"
    http.shutdown()


def test_prepare_writes_header_once_and_refuses_another_source(tmp_path):
    carpeta = tmp_path / "fuente" / "conjunto"
    banco.preparar(carpeta, FUENTE)
    banco.preparar(carpeta, FUENTE)
    fuente, archivos = banco.leer(carpeta)
    assert fuente == FUENTE
    assert archivos == []
    otra = banco.Fuente(**{**FUENTE.__dict__, "organismo": "Otro"})
    with pytest.raises(ProvenanceError, match="another source"):
        banco.preparar(carpeta, otra)


def test_download_writes_file_and_appends_record(tmp_path, servidor):
    carpeta = tmp_path / "c"
    banco.preparar(carpeta, FUENTE)
    archivo = banco.descargar(f"{servidor}/a.tif", carpeta, "a.tif", "2024", True, espera=0)
    assert (carpeta / "a.tif").read_bytes() == CONTENIDO
    assert archivo.sha256 == hashlib.sha256(CONTENIDO).hexdigest()
    assert archivo.bytes == len(CONTENIDO)
    assert not (carpeta / "a.tif.part").exists()
    _, archivos = banco.leer(carpeta)
    assert archivos == [archivo]
    assert banco.comprobar(carpeta) == [archivo]


def test_second_entry_is_appended_not_rewritten(tmp_path, servidor):
    carpeta = tmp_path / "c"
    banco.preparar(carpeta, FUENTE)
    banco.descargar(f"{servidor}/a.tif", carpeta, "a.tif", "2023", True, espera=0)
    antes = (carpeta / banco.PROCEDENCIA).read_bytes()
    banco.descargar(f"{servidor}/b.tif", carpeta, "b.tif", "2024", True, espera=0)
    despues = (carpeta / banco.PROCEDENCIA).read_bytes()
    assert despues.startswith(antes)
    assert [a.nombre for a in banco.leer(carpeta)[1]] == ["a.tif", "b.tif"]


def test_retries_server_errors(tmp_path, servidor):
    carpeta = tmp_path / "c"
    banco.preparar(carpeta, FUENTE)
    Servidor.fallos_restantes = 2
    archivo = banco.descargar(f"{servidor}/a.tif", carpeta, "a.tif", "2024", True, espera=0)
    assert archivo.bytes == len(CONTENIDO)


def test_gives_up_after_the_last_attempt(tmp_path, servidor):
    carpeta = tmp_path / "c"
    banco.preparar(carpeta, FUENTE)
    Servidor.fallos_restantes = 5
    with pytest.raises(SourceError, match="503"):
        banco.descargar(f"{servidor}/a.tif", carpeta, "a.tif", "2024", True, espera=0)
    assert banco.leer(carpeta)[1] == []


def test_client_error_fails_at_once(tmp_path, servidor):
    carpeta = tmp_path / "c"
    banco.preparar(carpeta, FUENTE)
    with pytest.raises(SourceError, match="404"):
        banco.descargar(f"{servidor}/no-existe.tif", carpeta, "x.tif", "2024", True, espera=0)
    assert Servidor.pedidos == 1


def test_cut_download_is_not_recorded(tmp_path, servidor):
    carpeta = tmp_path / "c"
    banco.preparar(carpeta, FUENTE)
    Servidor.cortar = True
    with pytest.raises(SourceError):
        banco.descargar(f"{servidor}/a.tif", carpeta, "a.tif", "2024", True, espera=0)
    assert not (carpeta / "a.tif").exists()
    assert banco.leer(carpeta)[1] == []


def test_refuses_overwrite_and_duplicate_names(tmp_path, servidor):
    carpeta = tmp_path / "c"
    banco.preparar(carpeta, FUENTE)
    banco.descargar(f"{servidor}/a.tif", carpeta, "a.tif", "2024", True, espera=0)
    with pytest.raises(SourceError, match="overwrite"):
        banco.descargar(f"{servidor}/a.tif", carpeta, "a.tif", "2024", True, espera=0)
    (carpeta / "a.tif").rename(carpeta / "a2.tif")
    with pytest.raises(ProvenanceError, match="already recorded"):
        banco.descargar(f"{servidor}/a.tif", carpeta, "a.tif", "2024", True, espera=0)


def test_download_needs_a_prepared_folder(tmp_path, servidor):
    with pytest.raises(ProvenanceError, match=r"no procedencia\.toml"):
        banco.descargar(f"{servidor}/a.tif", tmp_path, "a.tif", "2024", True, espera=0)


def test_remote_size(servidor):
    assert banco.tamano_remoto(f"{servidor}/a.tif", espera=0) == len(CONTENIDO)
    with pytest.raises(SourceError, match="404"):
        banco.tamano_remoto(f"{servidor}/no-existe.tif", espera=0)


def test_plain_http_is_refused(tmp_path):
    with pytest.raises(SourceError, match="https"):
        banco.tamano_remoto("http://ejemplo.org/a.tif")


def test_check_detects_changed_file(tmp_path, servidor):
    carpeta = tmp_path / "c"
    banco.preparar(carpeta, FUENTE)
    banco.descargar(f"{servidor}/a.tif", carpeta, "a.tif", "2024", True, espera=0)
    datos = bytearray((carpeta / "a.tif").read_bytes())
    datos[0] ^= 1
    (carpeta / "a.tif").write_bytes(bytes(datos))
    with pytest.raises(ProvenanceError, match="sha256"):
        banco.comprobar(carpeta)
    (carpeta / "a.tif").unlink()
    with pytest.raises(ProvenanceError, match="missing"):
        banco.comprobar(carpeta)


def test_remote_status_does_not_raise_on_client_errors(servidor):
    assert banco.estado_remoto(f"{servidor}/a.tif") == (200, len(CONTENIDO))
    assert banco.estado_remoto(f"{servidor}/no-existe.tif") == (404, None)


def test_register_a_file_already_in_the_folder(tmp_path):
    carpeta = tmp_path / "fuente" / "conjunto"
    banco.preparar(carpeta, FUENTE)
    (carpeta / "drive.zip").write_bytes(CONTENIDO)
    archivo = banco.registrar(carpeta, "drive.zip", "2013-2025", "earthengine:x", False)
    assert archivo.bytes == len(CONTENIDO)
    assert archivo.sha256 == hashlib.sha256(CONTENIDO).hexdigest()
    assert archivo.fecha_descarga[:4].isdigit() and "T" in archivo.fecha_descarga
    assert banco.zips(carpeta) == [carpeta / "drive.zip"]
    fijo = banco.registrar
    (carpeta / "b.txt").write_bytes(b"x")
    otro = fijo(carpeta, "b.txt", "2024", "https://x/b.txt", True, "2026-11-04T10:00:00-03:00")
    assert otro.fecha_descarga == "2026-11-04T10:00:00-03:00"
    assert banco.zips(carpeta) == [carpeta / "drive.zip"]


def test_register_refuses_missing_paths_and_repeats(tmp_path):
    carpeta = tmp_path / "f" / "c"
    banco.preparar(carpeta, FUENTE)
    with pytest.raises(ProvenanceError, match="file not found"):
        banco.registrar(carpeta, "nada.zip", "2024", "u", False)
    with pytest.raises(ProvenanceError, match="file name inside"):
        banco.registrar(carpeta, "../fuera.zip", "2024", "u", False)
    (carpeta / "a.zip").write_bytes(b"a")
    banco.registrar(carpeta, "a.zip", "2024", "u", False)
    with pytest.raises(ProvenanceError, match="already recorded"):
        banco.registrar(carpeta, "a.zip", "2024", "u", False)


def test_cli_registers(tmp_path, capsys):
    from nadir.__main__ import main

    carpeta = tmp_path / "f" / "c"
    banco.preparar(carpeta, FUENTE)
    (carpeta / "a.zip").write_bytes(b"abc")
    argumentos = ["banco", "registrar", "--carpeta", str(carpeta), "--archivo", "a.zip"]
    argumentos += ["--periodo", "2024", "--url", "earthengine:x", "--fecha", "2026-11-04"]
    assert main(argumentos) == 0
    assert "a.zip: 3 bytes" in capsys.readouterr().out
    _, archivos = banco.leer(carpeta)
    assert archivos[0].url_verificada is False
    assert archivos[0].fecha_descarga == "2026-11-04"
