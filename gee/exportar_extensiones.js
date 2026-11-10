/**
 * Exports for the extensions of the fire and land cover study (Nadir).
 *
 *   1. Land cover collection 1, years 2012 to 2022, for the robustness check against
 *      collection 2. Exported only if its grid is identical to the grid of collection 2
 *      (same CRS and same transform); otherwise the script stops: nothing is resampled.
 *   2. The seven regions and their municipalities, as GeoJSON, to assign pixels to regions
 *      and municipalities. They are used to add up areas only; they are not redistributed.
 *
 * Same area, tiling and conventions as exportar_mapbiomas_7_regiones.js:
 *   - 255 = outside the seven regions (no data in the GeoTIFF); 0 keeps its meaning.
 *   - uint8, native grid, tiles of 4096 x 4096 pixels, Google Drive.
 *
 * Use: paste into code.earthengine.google.com, press Run, read the Console, then start the
 * tasks one at a time from the Tasks tab. This script only creates the tasks.
 */

// ---------------------------------------------------------------------------
// 1. Configuration
// ---------------------------------------------------------------------------

var BASE = 'projects/mapbiomas-chile/assets/';
var COBERTURA_C2 = BASE + 'LULC/COLLECTION-02/CLASSIFICATIONS/classification-final/clasificacion-final-2';
var LIMITES = BASE + 'Territories_2026_07';
var REGIONES_ACTIVO = LIMITES + '/CHI_POLITICAL_LEVEL_2_REGIONES_SIMPLIFICADO_V2_UNION_GLACIARES_V3';

var REGIONES = [
  'Valparaíso',
  'Metropolitana de Santiago',
  "Libertador General Bernardo O'Higgins",
  'Maule',
  'Ñuble',
  'Biobío',
  'La Araucanía'
];

var CARPETA_DRIVE = 'nadir_mapbiomas';
var FUERA = 255;
var TESELA = 4096;
var ANIOS_C1 = [];
for (var a = 2012; a <= 2022; a++) ANIOS_C1.push('classification_' + a);

// ---------------------------------------------------------------------------
// 2. Find the assets (collection 1 and the municipalities) and print what was found
// ---------------------------------------------------------------------------

function listar(carpeta) {
  var salida = [];
  var sinPermiso = [];
  var pendientes = [carpeta];
  while (pendientes.length > 0) {
    var actual = pendientes.shift();
    var hijos;
    try {
      hijos = ee.data.listAssets(actual).assets || [];
    } catch (e) {
      sinPermiso.push(actual);
      continue;
    }
    hijos.forEach(function (h) {
      if (h.type === 'FOLDER') pendientes.push(h.name);
      else salida.push({id: h.id || h.name, tipo: h.type});
    });
  }
  if (sinPermiso.length > 0) print('Folders not readable (skipped):', sinPermiso);
  return salida;
}

var lulc = listar(BASE + 'LULC');
print('Assets under LULC:', lulc);
var candidatosC1 = lulc.filter(function (x) {
  return x.tipo === 'IMAGE' && /classif|clasific/i.test(x.id) && !/COLLECTION-02/i.test(x.id);
}).filter(function (x) {
  var bandas;
  try {
    bandas = ee.Image(x.id).bandNames().getInfo();
  } catch (e) {
    return false;
  }
  return ANIOS_C1.every(function (b) { return bandas.indexOf(b) !== -1; });
});
print('Collection 1 candidates with bands 2012-2022:', candidatosC1);
if (candidatosC1.length !== 1) {
  throw new Error('Expected exactly one collection 1 classification with bands 2012-2022, found ' +
    candidatosC1.length + '. Nothing was exported; keep the Console output.');
}
var COBERTURA_C1 = candidatosC1[0].id;

var limites = listar(LIMITES);
print("Assets under Territories_2026_07:", limites);
var candidatosComunas = limites.filter(function (x) {
  return x.tipo === 'TABLE' && /LEVEL_3|COMUNA/i.test(x.id);
});
print('Municipality candidates:', candidatosComunas);

// ---------------------------------------------------------------------------
// 3. Area and grid check
// ---------------------------------------------------------------------------

var regiones = ee.FeatureCollection(REGIONES_ACTIVO).filter(ee.Filter.inList('NAME', REGIONES));
var nombres = regiones.aggregate_array('NAME').getInfo();
if (nombres.length !== REGIONES.length) {
  throw new Error('Expected ' + REGIONES.length + ' regions, found ' + nombres.length + ': ' +
    JSON.stringify(nombres));
}
var area = regiones.geometry();
var rectangulo = area.bounds();

var proyC2 = ee.Image(COBERTURA_C2).projection().getInfo();
var proyC1 = ee.Image(COBERTURA_C1).projection().getInfo();
print('Collection 2 grid:', proyC2);
print('Collection 1 grid:', proyC1);
var mismaGrilla = proyC1.crs === proyC2.crs &&
  proyC1.transform.every(function (v, i) { return Math.abs(v - proyC2.transform[i]) < 1e-12; });
if (!mismaGrilla) {
  throw new Error('Collection 1 is not on the grid of collection 2. Nothing was exported: ' +
    'the method forbids resampling. Keep the Console output.');
}

// ---------------------------------------------------------------------------
// 4. Tasks
// ---------------------------------------------------------------------------

var nombreC1 = 'mbchile_cobertura_c1_2012_2022';
Export.image.toDrive({
  image: ee.Image(COBERTURA_C1).select(ANIOS_C1).clip(area).unmask(FUERA, false).toUint8(),
  description: nombreC1,
  folder: CARPETA_DRIVE,
  fileNamePrefix: nombreC1,
  region: rectangulo,
  crs: proyC1.crs,
  crsTransform: proyC1.transform,
  maxPixels: 1e13,
  fileDimensions: TESELA,
  fileFormat: 'GeoTIFF',
  formatOptions: {noData: FUERA}
});

Export.table.toDrive({
  collection: regiones,
  description: 'mbchile_regiones_7',
  folder: CARPETA_DRIVE,
  fileNamePrefix: 'mbchile_regiones_7',
  fileFormat: 'GeoJSON'
});

if (candidatosComunas.length === 1) {
  var comunas = ee.FeatureCollection(candidatosComunas[0].id).filterBounds(area);
  print('Municipalities touching the seven regions:', comunas.size());
  Export.table.toDrive({
    collection: comunas,
    description: 'mbchile_comunas_7_regiones',
    folder: CARPETA_DRIVE,
    fileNamePrefix: 'mbchile_comunas_7_regiones',
    fileFormat: 'GeoJSON'
  });
} else {
  print('Municipalities: ' + candidatosComunas.length + ' candidates; no task created. ' +
    'Keep the Console output.');
}

print('Collection 1 asset: ' + COBERTURA_C1);
print('Done. Start the tasks one at a time from the Tasks tab, collection 1 first.');
