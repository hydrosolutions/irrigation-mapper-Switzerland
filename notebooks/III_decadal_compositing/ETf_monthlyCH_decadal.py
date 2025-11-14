import ee
import re

# Initialize Earth Engine
# Re-authenticate and Initialize Earth Engine with ee-sahellakes project
try:
    # Try to authenticate with the ee-sahellakes project
    ee.Authenticate()
    ee.Initialize(project='ee-sahellakes')
except Exception as e:
    # Fallback to default initialization
    try:
        ee.Initialize()
    except Exception as e2:
        raise

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================
##note: Landsat data downloaded from https://espa.cr.usgs.gov/ordering/new/

CONFIG = {
    # Asset paths
    "paths": {
        "dailyETF": "projects/thurgau-irrigation/assets/ETlandsatmerged_ETF",
        "outputIC": "projects/thurgau-irrigation/assets/Zuerich/ET_products/decadal_Landsat_30m_ETF",
        "cantons": "projects/thurgau-irrigation/assets/GIS/Kantone_simplified100m",
    },
    # Processing parameters
    "targetCanton": "Zürich",
    "aggregationStat": "mean",
    # Time range
    "firstYear": 2018,
    "lastYear": 2022,
}

# =============================================================================
# INITIALIZATION
# =============================================================================

# Load required modules
gee_codes = ee.FeatureCollection("users/hydrosolutions/public_functions:SedimentBalance_T1L2")
Composites = ee.FeatureCollection("users/hydrosolutions/public_functions:composites.js")

# Load administrative boundaries
adm1_units = ee.FeatureCollection(CONFIG["paths"]["cantons"])

# Load ETF collections
et_daily_collection_all = ee.ImageCollection(CONFIG["paths"]["dailyETF"])

print("Available cantons:", adm1_units.aggregate_array("NAME").distinct().getInfo())
print("ETF daily collection size:", et_daily_collection_all.size().getInfo())

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def pad(num, size):
    """Pad a number with leading zeros"""
    num = str(num)
    while len(num) < size:
        num = "0" + num
    return num

def generate_monthly_intervals(start_year, start_month, end_year, end_month):
    """Generate monthly intervals with 3 periods per month (decadal)"""
    def add_month_intervals(month_index, prev_list):
        year = ee.Number(month_index).divide(12).floor()
        month = ee.Number(month_index).mod(12).add(1)

        new_intervals = ee.List([
            ee.List([ee.Date.fromYMD(year, month, 1),
                     ee.Date.fromYMD(year, month, 11)]),
            ee.List([ee.Date.fromYMD(year, month, 11),
                     ee.Date.fromYMD(year, month, 21)]),
            ee.List([ee.Date.fromYMD(year, month, 21),
                     ee.Algorithms.If(
                         month.eq(12),
                         ee.Date.fromYMD(year.add(1), 1, 1),
                         ee.Date.fromYMD(year, month.add(1), 1)
                     )])
        ])
        return ee.List(prev_list).cat(new_intervals)

    start_index = start_year * 12 + (start_month - 1)
    end_index = end_year * 12 + (end_month - 1)
    month_sequence = ee.List.sequence(start_index, end_index)

    return ee.List(month_sequence.iterate(add_month_intervals, ee.List([])))

# =============================================================================
# LANDSAT DATA PROCESSING
# =============================================================================
def cloudscore_L8_T1L2(image, area_of_interest):
    """Apply cloud scoring to Landsat 8/9 SR Tier 1/2 L2"""
    qaMask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
    mask = qaMask
    
    image2 = ee.Image(100).rename('cloud').where(qaMask.eq(1), 0)
    image3 = image2.updateMask(image.select('QA_PIXEL').neq(0))
    
    # Scale optical and thermal bands
    opticalBands = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
    thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
    
    # Cloud cover fractions
    cloudPixels = ee.Number(
        image2.select("cloud").reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=area_of_interest,
            scale=100,
            tileScale=2
        ).get("cloud")
    )
    cloudPixels2 = ee.Number(
        image3.select("cloud").reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=area_of_interest,
            scale=100,
            tileScale=2
        ).get("cloud")
    )
    
    return (image
            .addBands(opticalBands, overwrite=True)
            .addBands(thermalBands, overwrite=True)
            .updateMask(image2.select(['cloud']).lt(100))
            .addBands(image2)
            .addBands(mask.rename('mask'))
            .set({'nodata_cover': cloudPixels,
                  'cloud_cover': cloudPixels2,
                  'SENSING_TIME': ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'),
                  'SATELLITE': 'LANDSAT_8'})
           )


def cloudscore_L7_T1L2(image, area_of_interest):
    """Apply cloud scoring to Landsat 7 SR Tier 1/2 L2"""
    qaMask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
    mask = qaMask
    
    image2 = ee.Image(100).rename('cloud').where(qaMask.eq(1), 0)
    image3 = image2.updateMask(image.select('QA_PIXEL').neq(0))
    
    # Scale optical and thermal bands
    opticalBands = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
    thermalBand = image.select('ST_B6').multiply(0.00341802).add(149.0)
    
    # Cloud cover fractions
    cloudPixels = ee.Number(
        image2.select("cloud").reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=area_of_interest,
            scale=100,
            tileScale=2
        ).get("cloud")
    )
    cloudPixels2 = ee.Number(
        image3.select("cloud").reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=area_of_interest,
            scale=100,
            tileScale=2
        ).get("cloud")
    )
    
    return (image
            .addBands(opticalBands, overwrite=True)
            .addBands(thermalBand, overwrite=True)
            .updateMask(image2.select(['cloud']).lt(100))
            .addBands(image2)
            .addBands(mask.rename('mask'))
            .set({'nodata_cover': cloudPixels,
                  'cloud_cover': cloudPixels2,
                  'SENSING_TIME': ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'),
                  'SATELLITE': 'LANDSAT_7'})
           )

def aggregateStack(masked_collection, band_list, time_interval, options=None):
    """
    Generate a temporally-aggregated image for a given time interval.

    Args:
        masked_collection (ee.ImageCollection): Input collection (e.g. cloud-masked).
        band_list (list): List of band names.
        time_interval (ee.List): [startDate, endDate] as ee.Date objects.
        options (dict): Options:
            - 'band_name': str, default 'NDVI'
            - 'agg_type': str, one of ['geomedian', 'mean', 'max', 'min', 'median']

    Returns:
        ee.Image: Aggregated image with system:time_start set to interval center.
    """
    if options is None:
        options = {}

    band_name = options.get("band_name", "NDVI")
    agg_type = options.get("agg_type", "median")

    time_interval = ee.List(time_interval)
    agg_interval = ee.Date(time_interval.get(1)).difference(time_interval.get(0), "day")

    timestamp = {
        "system:time_start": ee.Date(time_interval.get(0))
        .advance(ee.Number(agg_interval.divide(2)).ceil(), "day")
        .millis()
    }

    collection_filtered = masked_collection.filterDate(time_interval.get(0), time_interval.get(1))

    def empty_image():
        """Create empty image with same bands if no observations."""
        return ee.Image(
            ee.List(band_list[1:]).iterate(
                lambda band, stack: ee.Image(stack).addBands(ee.Image()),
                ee.Image()
            )
        ).rename(band_list).set(timestamp).toFloat()

    if agg_type == "geomedian":
        agg_image = ee.Algorithms.If(
            collection_filtered.size().gt(0),
            collection_filtered.select(band_list)
                .reduce(ee.Reducer.geometricMedian(len(band_list)), 1)
                .rename(band_list).set(timestamp),
            empty_image()
        )

    elif agg_type == "mean":
        agg_image = ee.Algorithms.If(
            collection_filtered.size().gt(0),
            collection_filtered.select(band_list)
                .reduce(ee.Reducer.mean(), 1)
                .rename(band_list).set(timestamp),
            empty_image()
        )

    elif agg_type == "max":
        agg_image = ee.Algorithms.If(
            collection_filtered.size().gt(0),
            collection_filtered.select(band_list)
                .reduce(ee.Reducer.max(), 1)
                .rename(band_list).set(timestamp),
            empty_image()
        )

    elif agg_type == "min":
        agg_image = ee.Algorithms.If(
            collection_filtered.size().gt(0),
            collection_filtered.select(band_list)
                .reduce(ee.Reducer.min(), 1)
                .rename(band_list).set(timestamp),
            empty_image()
        )

    else:  # default to median if nothing matches
        agg_image = ee.Algorithms.If(
            collection_filtered.size().gt(0),
            collection_filtered.select(band_list)
                .reduce(ee.Reducer.median(), 1)
                .rename(band_list).set(timestamp),
            empty_image()
        )

    return ee.Image(agg_image)

def harmonizedTS(masked_collection, band_list, time_intervals, options=None):
    """
    Create a harmonized time series from a masked collection over given time intervals.

    Args:
        masked_collection (ee.ImageCollection): Input collection (e.g. cloud-masked).
        band_list (list): List of bands to aggregate.
        time_intervals (ee.List): List of [start, end] ee.Date intervals.
        options (dict): Options, accepts:
            - 'band_name' (str): name of output band (default = 'NDVI')
            - 'agg_type' (str): aggregation type: mean, median, min, max (default = 'median')

    Returns:
        ee.ImageCollection: Time series of aggregated composites.
    """

    if options is None:
        options = {}

    band_name = options.get("band_name", "NDVI")
    agg_type = options.get("agg_type", "median")

    def _stackBands(time_interval, stack):
        outputs = aggregateStack(
            masked_collection,
            band_list,
            ee.List(time_interval),
            {"agg_type": agg_type, "band_name": band_name}
        )
        return ee.List(stack).add(ee.Image(outputs))

    stack = ee.List([])

    agg_stack = ee.List(time_intervals).iterate(_stackBands, stack)

    return ee.ImageCollection(ee.List(agg_stack)).sort("system:time_start")


def load_landsat_collections(aoi, year):
    """Load and process Landsat collections with cloud masking"""
    l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
          .merge(ee.ImageCollection("LANDSAT/LC08/C02/T2_L2"))
          .filterBounds(aoi)
          .filter(ee.Filter.calendarRange(year, year, "year")))

    l9 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
          .filterBounds(aoi)
          .filter(ee.Filter.calendarRange(year, year, "year")))

    l7 = (ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
          .merge(ee.ImageCollection("LANDSAT/LE07/C02/T2_L2"))
          .filterBounds(aoi)
          .filter(ee.Filter.calendarRange(year, year, "year")))

    merged_landsat = (l8.map(lambda img: cloudscore_L8_T1L2(ee.Image(img), aoi))
                        .merge(l9.map(lambda img: cloudscore_L8_T1L2(ee.Image(img), aoi))))

    if year < 2022:
        merged_landsat = merged_landsat.merge(
            l7.map(lambda img: cloudscore_L7_T1L2(ee.Image(img), aoi))
        )

    return merged_landsat


def match_etf_with_landsat(etf_collection, landsat_collection, aoi):
    """Match ETF images with Landsat cloud masks"""
    def process(img):
        img = ee.Image(img)
        date = ee.Date.fromYMD(img.get("year"), img.get("month"), img.get("day"))
        path = img.get("WRS_PATH")
        row = img.get("WRS_ROW")
        satId = ee.String(img.get("SPACECRAFT_ID"))

        satellite = ee.String(
            ee.Algorithms.If(
                satId.compareTo("LC08").eq(0), "LANDSAT_8",
                ee.Algorithms.If(satId.compareTo("LC09").eq(0), "LANDSAT_9", "LANDSAT_7")
            )
        )

        landsat = (landsat_collection
                   .filterDate(date, date.advance(1, "day"))
                   .filter(ee.Filter.And(
                       ee.Filter.eq("SPACECRAFT_ID", satellite),
                       ee.Filter.eq("WRS_PATH", path),
                       ee.Filter.eq("WRS_ROW", row)
                   )))

        has_landsat = landsat.size().gt(0)

        return ee.Image(
            ee.Algorithms.If(
                has_landsat,
                img.rename("ETF")
                   .set("system:time_start", date)
                   .updateMask(ee.Image(landsat.first()).select("cloud").eq(0))
                   .copyProperties(landsat.first())
                   .set("matched", 1),
                img.set("system:time_start", date).set("matched", 0)
            )
        ).clip(aoi)

    return ee.ImageCollection(etf_collection.map(process)).filter(ee.Filter.eq("matched", 1))


def create_decadal_composites(etf_collection, time_intervals, aoi, aggregation_stat):
    """Create decadal composites from daily ETF data"""
    etf_composites = harmonizedTS(etf_collection, ["ETF"], time_intervals, {"agg_type": aggregation_stat})
    composites_list = etf_composites.toList(36)

    def make_img(x):
        interval = ee.List(time_intervals.get(ee.Number(x)))
        days = ee.Date(interval.get(1)).difference(ee.Date(interval.get(0)), "day")
        n = etf_collection.filterDate(ee.Date(interval.get(0)), ee.Date(interval.get(1))).size()
        cc = etf_collection.filterDate(ee.Date(interval.get(0)), ee.Date(interval.get(1))) \
            .aggregate_array("cloud_cover").reduce(ee.Reducer.mean())

        return (ee.Image(composites_list.get(ee.Number(x)))
                .set("cloud_cover", cc)
                .set("n", n)
                .set("days", days)
                .set("system:time_start", ee.Date(interval.get(0)).millis()))

    return ee.ImageCollection(ee.List.sequence(0, ee.Number(time_intervals.length()).subtract(1)).map(make_img))


def convert_etf_values(etf_collection, aoi):
    """Convert ETF values to appropriate format"""
    def process(img):
        img = ee.Image(img)
        return (img.clip(aoi)
                .toShort()
                .copyProperties(img, ["system:time_start", "days", "n", "cloud_cover"])
                .set("SENSING_TIME", ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")))

    return ee.ImageCollection(etf_collection.map(process))


def process_etf_decadal(aoi, year):
    """Main function to process ETF data for decadal composites"""
    aoi = ee.Geometry(aoi)
    time_intervals = generate_monthly_intervals(year, 1, year, 12)

    merged_landsat = load_landsat_collections(aoi, year)

    etf_daily_collection = (et_daily_collection_all
                            .filterBounds(aoi)
                            .filter(ee.Filter.eq("year", year)))

    etf_daily_collection = match_etf_with_landsat(etf_daily_collection, merged_landsat, aoi)
    etf_daily_collection = etf_daily_collection.filter(ee.Filter.lt("cloud_cover", 40))

    etf_decadal_collection = create_decadal_composites(etf_daily_collection, time_intervals, aoi, CONFIG["aggregationStat"])
    etf_decadal_collection = convert_etf_values(etf_decadal_collection, aoi)

    return etf_decadal_collection


def export_etf_decadal(canton_name, start_year, end_year):
    """Export decadal ETF composites for a specific canton and year range"""
    aoi = adm1_units.filter(ee.Filter.eq("NAME", canton_name)).geometry().buffer(200,100)
    clean_canton_name = re.sub("ü", "u", canton_name.replace(" ", ""))

    export_list = []

    for yr in range(start_year, end_year + 1):
        etf_decadal_collection = process_etf_decadal(aoi, yr)
        etf_list = etf_decadal_collection.toList(36)

        for xx in range(12, 27):
            img = ee.Image(etf_list.get(xx))
            n = ee.Number(img.get("n"))

            img = ee.Image(ee.Algorithms.If(
                n.eq(0),
                ee.Image().rename("ETF").copyProperties(img).set("system:time_start", img.get("system:time_start")),
                img
            ))

            mm = (xx // 3) + 1
            decade = (xx % 3) + 1

            image_name = f"ETF_Landsat_decadal_{clean_canton_name}_{yr}{pad(mm,2)}_{decade}"

            task = ee.batch.Export.image.toAsset(
                image=img.toInt().clip(aoi).set("Region", canton_name),
                description=image_name,
                assetId=f"{CONFIG['paths']['outputIC']}/{image_name}",
                scale=30,
                maxPixels=1e13,
                region=aoi
            )
            task.start()

            export_list.append({"name": image_name, "date": ee.Date(img.get("system:time_start"))})

    return export_list


# =============================================================================
# RUN EXPORT
# =============================================================================

print("Starting export for:", CONFIG["targetCanton"])
export_results = export_etf_decadal(CONFIG["targetCanton"], CONFIG["firstYear"], CONFIG["lastYear"])
print("Export tasks created:", len(export_results))
