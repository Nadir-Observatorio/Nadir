/**
 * Export of the 30 m Global Annual Burned Area Maps (GABAM; Long et al., 2019) for the seven
 * regions of the fire and land cover study (Nadir), years 2013 to 2021, as the fire source of
 * H3' (a definition of fire that does not go through the thematic filter of MapBiomas Fire).
 *
 * Source: Earth Engine community catalog, projects/sat-io/open-datasets/GABAM (CC0 1.0 in the
 * catalog; data citation Long, Zhang and He, 2021, doi:10.7910/DVN/3CTMKP).
 *
 *   - One task per year, on the native grid of GABAM (0.00025 degrees, EPSG:4326): nothing is
 *     resampled here. Pixels are read later on the land cover grid by pixel centre.
 *   - uint8: GABAM value inside the seven regions with 0 where GABAM has no burned pixel;
 *     255 outside the seven regions (no data in the GeoTIFF).
 *   - Tiles of 4096 x 4096 pixels, Google Drive folder nadir_gabam.
 *
 * Use: paste into code.earthengine.google.com, press Run, read the Console, then start the
 * tasks from the Tasks tab. The script stops before creating tasks if a year is missing or the
 * grid of the year is not the expected one.
 */

var GABAM = 'projects/sat-io/open-datasets/GABAM';
var LIMITES = 'projects/mapbiomas-chile/assets/Territories_2026_07';
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
var ANIOS = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021];
var CARPETA_DRIVE = 'nadir_gabam';
var FUERA = 255;
var TESELA = 4096;
var PASO = 0.00025;

var regiones = ee.FeatureCollection(REGIONES_ACTIVO).filter(ee.Filter.inList('NAME', REGIONES));
var nombres = regiones.aggregate_array('NAME').getInfo();
if (nombres.length !== REGIONES.length) {
  throw new Error('Expected ' + REGIONES.length + ' regions, found ' + nombres.length + ': ' +
    JSON.stringify(nombres));
}
var area = regiones.geometry();
var rectangulo = area.bounds();

var coleccion = ee.ImageCollection(GABAM).filterBounds(rectangulo);
print('GABAM images over the seven regions:', coleccion.size());

var porAnio = ANIOS.map(function (anio) {
  var inicio = ee.Date.fromYMD(anio, 1, 1);
  var imagenes = coleccion.filterDate(inicio, inicio.advance(1, 'year'));
  var n = imagenes.size().getInfo();
  if (n === 0) {
    throw new Error('GABAM has no image for ' + anio + ' over the seven regions. ' +
      'Nothing was exported; keep the Console output.');
  }
  var primera = ee.Image(imagenes.first());
  var proy = primera.projection().getInfo();
  var bandas = primera.bandNames().getInfo();
  print(anio + ': ' + n + ' tiles, bands ' + JSON.stringify(bandas) + ', grid', proy);
  if (proy.crs !== 'EPSG:4326' || Math.abs(proy.transform[0] - PASO) > 1e-12 ||
      Math.abs(proy.transform[4] + PASO) > 1e-12) {
    throw new Error('GABAM ' + anio + ' is not on a 0.00025 degree EPSG:4326 grid. ' +
      'Nothing was exported; keep the Console output.');
  }
  return {anio: anio, imagenes: imagenes, proy: proy, banda: bandas[0]};
});

porAnio.forEach(function (x) {
  var valores = x.imagenes.select(x.banda).mosaic()
    .reduceRegion({reducer: ee.Reducer.frequencyHistogram(), geometry: rectangulo,
                   crs: x.proy.crs, crsTransform: x.proy.transform, maxPixels: 1e13,
                   bestEffort: true});
  print(x.anio + ' values (best effort, for reading only):', valores);
});

porAnio.forEach(function (x) {
  var nombre = 'gabam_7_regiones_' + x.anio;
  var imagen = x.imagenes.select(x.banda).mosaic()
    .unmask(0).clip(area).unmask(FUERA, false).toUint8().rename('quemado_' + x.anio);
  Export.image.toDrive({
    image: imagen,
    description: nombre,
    folder: CARPETA_DRIVE,
    fileNamePrefix: nombre,
    region: rectangulo,
    crs: x.proy.crs,
    crsTransform: x.proy.transform,
    maxPixels: 1e13,
    fileDimensions: TESELA,
    fileFormat: 'GeoTIFF',
    formatOptions: {noData: FUERA}
  });
});

print('Done. Start the nine tasks from the Tasks tab; the files go to Drive/' + CARPETA_DRIVE);
