import pytest

from nadir.errors import LicenseError
from nadir.licencias import LICENCIAS, heredada, licencia


def test_unknown_licence_is_refused():
    with pytest.raises(LicenseError, match="not accepted"):
        licencia("CC-BY-NC-4.0")


def test_single_input_keeps_its_licence():
    assert heredada(["CC-BY-4.0"]).codigo == "CC-BY-4.0"


def test_most_restrictive_wins():
    assert heredada(["CC0-1.0", "CC-BY-4.0"]).codigo == "CC-BY-4.0"
    assert heredada(["CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0"]).codigo == "CC-BY-SA-4.0"


def test_public_domain_inputs_stay_public_domain():
    assert heredada(["CC0-1.0", "PDDL-1.0"]).rango == 0


def test_mapbiomas_with_attribution_licence():
    resultado = heredada(["MapBiomas-Chile", "CC-BY-4.0"])
    assert resultado.codigo == "MapBiomas-Chile"
    assert resultado.no_comercial


def test_mapbiomas_cannot_be_combined_with_share_alike():
    with pytest.raises(LicenseError, match="non-commercial"):
        heredada(["MapBiomas-Chile", "CC-BY-SA-4.0"])


def test_empty_input_is_refused():
    with pytest.raises(LicenseError, match="at least one"):
        heredada([])


def test_every_licence_has_an_https_url():
    assert all(lic.url.startswith("https://") for lic in LICENCIAS.values())


def test_copernicus_dem_is_an_attribution_licence():
    copernicus = licencia("Copernicus-DEM-GLO-30")
    assert copernicus.rango == LICENCIAS["CC-BY-4.0"].rango
    assert not copernicus.comparte_igual
    assert not copernicus.no_comercial
    assert heredada(["Copernicus-DEM-GLO-30", "MapBiomas-Chile"]).codigo == "MapBiomas-Chile"
