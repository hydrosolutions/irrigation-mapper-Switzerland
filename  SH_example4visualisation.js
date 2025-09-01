var img=ee.Image('projects/thurgau-irrigation/assets/Schaffhausen/ET_products/decadal_Landsat_30m/ET_Landsat_decadal_Schaffhausen_202306_3')
Map.addLayer(img,{min:0,max:40,palette:'white,yellow,red,green'},'ETA 2023',false)

var img2=ee.Image('projects/thurgau-irrigation/assets/Schaffhausen/ET_products/decadal_Landsat_30m_ETF/ETF_Landsat_decadal_Schaffhausen_202306_1')
Map.addLayer(img2.divide(10000).divide(0.851851852),{min:0.2,max:1,palette:'white,yellow,red,green'},'ETF 2023a',false)

var img2=ee.Image('projects/thurgau-irrigation/assets/Schaffhausen/ET_products/decadal_Landsat_30m_ETF/ETF_Landsat_decadal_Schaffhausen_202306_2')
Map.addLayer(img2.divide(10000).divide(0.878787879),{min:0.2,max:1,palette:'white,yellow,red,green'},'ETF 2023b',false)

var img2=ee.Image('projects/thurgau-irrigation/assets/Schaffhausen/ET_products/decadal_Landsat_30m_ETF/ETF_Landsat_decadal_Schaffhausen_202306_3')
Map.addLayer(img2.divide(10000).divide(0.909090909),{min:0.2,max:1,palette:'white,yellow,red,green'},'ETF 2023c',false)


var rhiresD=ee.ImageCollection('projects/thurgau-irrigation/assets/Precipitation/RhiresD').filter(ee.Filter.eq('year',2023))
  .filter(ee.Filter.eq('month',6))
  
// Extract daily precipitation at the geometry
var dailyPrecip = rhiresD.map(function(img) {
  var value = img.reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: geometry,
    scale: 1000
  });
  var year = ee.Number(img.get('year'));
  var month = ee.Number(img.get('month'));
  var day = ee.Number(img.get('day'));
  var date = ee.Date.fromYMD(year, month, day);
  return ee.Feature(null, {
    'date': date.format('YYYY-MM-dd'),
    'precip': value.values().get(0)
  });
});

// Convert to a FeatureCollection
var dailyPrecipFC = ee.FeatureCollection(dailyPrecip);

// Print daily precipitation to the console
print('Daily precipitation at geometry:', dailyPrecipFC);

// Plot daily precipitation
var chart = ui.Chart.feature.byFeature(dailyPrecipFC, 'date', 'precip')
  .setChartType('LineChart')
  .setOptions({
    title: 'Daily Precipitation at Geometry',
    hAxis: {title: 'Date'},
    vAxis: {title: 'Precipitation (mm)'},
    lineWidth: 2,
    pointSize: 4
  });

print(chart);

var rhiresD_sum=ee.ImageCollection('projects/thurgau-irrigation/assets/Precipitation/RhiresD').filter(ee.Filter.eq('year',2023))
  .filter(ee.Filter.eq('month',6)).filter(ee.Filter.lte('day',20)).filter(ee.Filter.gt('day',10)).sum()

Map.addLayer(rhiresD_sum,{min:0,max:50,palette:'white,blue'},'rhiresD 2023')

var img3=img.divide(img2.divide(10000).divide(0.91))

Map.addLayer(img3,{min:0.4,max:0.9,palette:'white,yellow,red,green'},'ETA vs ETF 2023',false)

var precip=ee.Image('projects/thurgau-irrigation/assets/monthly_precip_CH/2023_06')


// Map.centerObject(img)
// Map.addLayer(precip,{min:0,max:40,palette:'white,blue'})



var img=ee.Image('projects/thurgau-irrigation/assets/Schaffhausen/ET_products/decadal_Landsat_30m/ET_Landsat_decadal_Schaffhausen_202406_3')
Map.addLayer(img,{min:0,max:40,palette:'white,yellow,red,green'},'ETA 2024',false)


var img2=ee.Image('projects/thurgau-irrigation/assets/Schaffhausen/ET_products/decadal_Landsat_30m_ETF/ETF_Landsat_decadal_Schaffhausen_202406_1')
Map.addLayer(img2.divide(10000).divide(0.851851852),{min:0.2,max:1,palette:'white,yellow,red,green'},'ETF 2024a',false)

var img2=ee.Image('projects/thurgau-irrigation/assets/Schaffhausen/ET_products/decadal_Landsat_30m_ETF/ETF_Landsat_decadal_Schaffhausen_202406_2')
Map.addLayer(img2.divide(10000).divide(0.878787879),{min:0.2,max:1,palette:'white,yellow,red,green'},'ETF 2024b',false)

var img2=ee.Image('projects/thurgau-irrigation/assets/Schaffhausen/ET_products/decadal_Landsat_30m_ETF/ETF_Landsat_decadal_Schaffhausen_202406_3')
Map.addLayer(img2.divide(10000).divide(0.909090909),{min:0.2,max:1,palette:'white,yellow,red,green'},'ETF 2024c',false)

var rhiresD=ee.ImageCollection('projects/thurgau-irrigation/assets/Precipitation/RhiresD').filter(ee.Filter.eq('year',2024))
  .filter(ee.Filter.eq('month',6)).filter(ee.Filter.lte('day',20)).filter(ee.Filter.gt('day',10)).sum()

Map.addLayer(rhiresD,{min:0,max:50,palette:'white,blue'},'rhiresD 2024')

var rhiresD=ee.ImageCollection('projects/thurgau-irrigation/assets/Precipitation/RhiresD').filter(ee.Filter.eq('year',2024))
  .filter(ee.Filter.eq('month',6))
  
// Extract daily precipitation at the geometry
var dailyPrecip = rhiresD.map(function(img) {
  var value = img.reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: geometry,
    scale: 1000
  });
  var year = ee.Number(img.get('year'));
  var month = ee.Number(img.get('month'));
  var day = ee.Number(img.get('day'));
  var date = ee.Date.fromYMD(year, month, day);
  return ee.Feature(null, {
    'date': date.format('YYYY-MM-dd'),
    'precip': value.values().get(0)
  });
});

// Convert to a FeatureCollection
var dailyPrecipFC = ee.FeatureCollection(dailyPrecip);

// Print daily precipitation to the console
print('Daily precipitation at geometry:', dailyPrecipFC);

// Plot daily precipitation
var chart = ui.Chart.feature.byFeature(dailyPrecipFC, 'date', 'precip')
  .setChartType('LineChart')
  .setOptions({
    title: 'Daily Precipitation at Geometry',
    hAxis: {title: 'Date'},
    vAxis: {title: 'Precipitation (mm)'},
    lineWidth: 2,
    pointSize: 4
  });

print(chart);
var img3=img.divide(img2.divide(10000).divide(0.91))

Map.addLayer(img3,{min:0.4,max:0.9,palette:'white,yellow,red,green'},'ETA vs ETF 2024',false)

var kcs=ee.FeatureCollection('projects/thurgau-irrigation/assets/FribourgAndVaud/ETc_WAPOR/Kc_Pasture_Broye_Aquastat')
print(kcs.filter(ee.Filter.eq('Decade',1)).sort('Month'),kcs.filter(ee.Filter.eq('Decade',2)).sort('Month'),kcs.filter(ee.Filter.eq('Decade',3)).sort('Month'))