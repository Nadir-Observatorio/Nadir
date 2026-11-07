/**
 * Exportación de MapBiomas Chile a Google Drive para el estudio fuego-cobertura.
 *
 * Alcance: las 7 regiones que cubre MapBiomas Fuego Chile, Colección 1
 * (Valparaíso a La Araucanía).
 *   1. Cobertura c2, activo clasificacion-final-2, bandas 2012 a 2024.
 *   2. Área quemada anual c1, bandas 2013 a 2025 (1 = quemado).
 *   3. Área quemada mensual c1, bandas 2013 a 2025 (valor = mes de quema, 1 a 12).
 *
 * Cada producto sale en su proyección y grilla nativas (EPSG:4326, transformación del
 * activo), sin reproyectar. Todo en uint8.
 *   - 255 = fuera de las 7 regiones (marcado como sin dato en el GeoTIFF).
 *   - Cobertura: 0 conserva su significado original en MapBiomas (sin dato u otro).
 *   - Fuego: 0 = no quemado dentro de las 7 regiones.
 * Drive parte cada producto en teselas de 4096 x 4096 píxeles.
 *
 * Uso: pegar en code.earthengine.google.com, presionar Run, revisar la Console y luego
 * lanzar las tareas desde la pestaña Tasks DE A UNA (ver instrucciones en el chat).
 * Este script no exporta nada por sí solo: solo crea las tareas.
 */

// ---------------------------------------------------------------------------
// 1. Configuración
// ---------------------------------------------------------------------------

var BASE = 'projects/mapbiomas-chile/assets/';

var ACTIVOS = {
  cobertura: BASE + 'LULC/COLLECTION-02/CLASSIFICATIONS/classification-final/clasificacion-final-2',
  fuegoAnual: BASE + 'FIRE/COLLECTION1/FINAL_PRODUCTS/mapbiomas_chile_fire_collection1_annual_burned_v1',
  fuegoMensual: BASE + 'FIRE/COLLECTION1/FINAL_PRODUCTS/mapbiomas_chile_fire_collection1_monthly_burned_v1',
  regiones: BASE + 'Territories_2026_07/CHI_POLITICAL_LEVEL_2_REGIONES_SIMPLIFICADO_V2_UNION_GLACIARES_V3'
};

var REGIONES = [
  'Valparaíso',
  'Metropolitana de Santiago',
  "Libertador General Bernardo O'Higgins",
  'Maule',
  'Ñuble',
  'Biobío',
  'La Araucanía'
];

var CARPETA_DRIVE = 'rastro_mapbiomas';
var FUERA = 255;          // fuera de las 7 regiones
var TESELA = 4096;        // lado de cada archivo en píxeles (múltiplo de 256)

function rango(desde, hasta) {
  var lista = [];
  for (var anio = desde; anio <= hasta; anio++) lista.push(anio);
  return lista;
}

var PRODUCTOS = [
  {
    clave: 'cobertura',
    nombre: 'mbchile_cobertura_c2_final2_2012_2024',
    bandas: rango(2012, 2024).map(function (a) { return 'classification_' + a; }),
    fondo: null           // sin relleno: se respeta el 0 del activo
  },
  {
    clave: 'fuegoAnual',
    nombre: 'mbchile_fuego_c1_anual_2013_2025',
    bandas: rango(2013, 2025).map(function (a) { return 'burned_area_' + a; }),
    fondo: 0              // lo enmascarado dentro de las regiones = no quemado
  },
  {
    clave: 'fuegoMensual',
    nombre: 'mbchile_fuego_c1_mensual_2013_2025',
    bandas: rango(2013, 2025).map(function (a) { return 'burned_monthly_' + a; }),
    fondo: 0
  }
];

// ---------------------------------------------------------------------------
// 2. Área de exportación
// ---------------------------------------------------------------------------

var todas = ee.FeatureCollection(ACTIVOS.regiones);
var elegidas = todas.filter(ee.Filter.inList('NAME', REGIONES));

var encontradas = elegidas.aggregate_array('NAME').getInfo();
if (encontradas.length !== REGIONES.length) {
  throw new Error('Se esperaban ' + REGIONES.length + ' regiones y se encontraron ' +
    encontradas.length + ': ' + JSON.stringify(encontradas));
}
print('Regiones encontradas (' + encontradas.length + '): ' + encontradas.join(', '));

var area = elegidas.geometry();
var rectangulo = area.bounds();
Map.addLayer(elegidas, {color: 'ff7f00'}, '7 regiones');
Map.centerObject(elegidas, 6);

// ---------------------------------------------------------------------------
// 3. Preparación y tareas
// ---------------------------------------------------------------------------

function preparar(producto) {
  var imagen = ee.Image(ACTIVOS[producto.clave]).select(producto.bandas);
  if (producto.fondo !== null) imagen = imagen.unmask(producto.fondo);
  return imagen.clip(area).unmask(FUERA, false).toUint8();
}

function revisar(producto) {
  var original = ee.Image(ACTIVOS[producto.clave]);
  var disponibles = original.bandNames().getInfo();
  var faltan = producto.bandas.filter(function (b) { return disponibles.indexOf(b) === -1; });
  if (faltan.length > 0) {
    throw new Error(producto.nombre + ': el activo no trae ' + faltan.join(', '));
  }
  var proy = original.projection().getInfo();
  if (proy.crs !== 'EPSG:4326') {
    throw new Error(producto.nombre + ': proyección inesperada ' + proy.crs);
  }
  return proy;
}

function tamano(proy) {
  var esquinas = rectangulo.coordinates().getInfo()[0];
  var lons = esquinas.map(function (p) { return p[0]; });
  var lats = esquinas.map(function (p) { return p[1]; });
  var ancho = Math.ceil((Math.max.apply(null, lons) - Math.min.apply(null, lons)) / Math.abs(proy.transform[0]));
  var alto = Math.ceil((Math.max.apply(null, lats) - Math.min.apply(null, lats)) / Math.abs(proy.transform[4]));
  return {ancho: ancho, alto: alto};
}

PRODUCTOS.forEach(function (producto) {
  var proy = revisar(producto);
  var dim = tamano(proy);
  var bytes = dim.ancho * dim.alto * producto.bandas.length;
  var teselas = Math.ceil(dim.ancho / TESELA) * Math.ceil(dim.alto / TESELA);
  print(producto.nombre + ' | bandas: ' + producto.bandas.length +
    ' | ' + dim.ancho + ' x ' + dim.alto + ' píxeles' +
    ' | sin comprimir: ' + (bytes / 1e9).toFixed(1) + ' GB' +
    ' | archivos esperados: ' + teselas +
    ' | transformación: ' + JSON.stringify(proy.transform));

  Export.image.toDrive({
    image: preparar(producto),
    description: producto.nombre,
    folder: CARPETA_DRIVE,
    fileNamePrefix: producto.nombre,
    region: rectangulo,
    crs: proy.crs,
    crsTransform: proy.transform,
    maxPixels: 1e13,
    fileDimensions: TESELA,
    fileFormat: 'GeoTIFF',
    formatOptions: {noData: FUERA}
  });
});

print('Listo. Abre la pestaña Tasks y lanza las tareas de a una, empezando por fuego anual.');
