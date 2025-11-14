from typing import List, Tuple, Set
import ee
import matplotlib.pyplot as plt
import numpy as np
from src.et_green.compute_et_green import compute_et_green, compute_et_green_std
from src.et_green.filter_nutzungsflaechen import (
    get_crops_to_exclude,
    get_rainfed_reference_crops,
    create_crop_filters,
    filter_crops,
    add_double_cropping_info,#add_ALL_double_cropping_info,
    get_unique_nutzung,
)
from utils.ee_utils import back_to_int, export_image_to_asset, normalize_string_server,normalize_string_client

def get_time_step_pattern(date: ee.Date, time_step_type: str) -> str:
    """
    Get formatted time step pattern from a date based on type.

    Args:
        date (ee.Date): The date to process
        time_step_type (str): Either 'dekadal' or 'monthly'

    Returns:
        str: Formatted time step pattern (e.g. '04_D1' for dekadal or '04' for monthly)

    Raises:
        ValueError: If time_step_type is neither 'dekadal' nor 'monthly'
    """
    if time_step_type not in ["dekadal", "monthly"]:
        raise ValueError("time_step_type must be either 'dekadal' or 'monthly'")

    # Add 1 to month since GEE uses 0-based months
    month = date.get("month").getInfo()
    month_str = f"{month:02d}"

    if time_step_type == "monthly":
        return month_str

    # For dekadal, determine which 10-day period
    day = date.get("day").getInfo()
    dekadal = ((day - 1) // 10) + 1
    return f"{month_str}_D{dekadal}"


def normalize_feature(feature: ee.Feature, property: str = "nutzung") -> ee.Feature:
    """Normalizes a property's string value in an Earth Engine Feature by replacing special characters.

    Adds a new property with suffix '_normalized' containing the normalized string value.
    For example, if property is "nutzung", creates "nutzung_normalized".

    Args:
        feature (ee.Feature): The Earth Engine Feature containing the property to normalize.
        property (str, optional): Name of the property to normalize. Defaults to "nutzung".

    Returns:
        ee.Feature: The input feature with an additional normalized property.
    """
    prop_value = ee.String(feature.get(property))

    normalized_prop_name = ee.String(property).cat("_normalized")

    normalized = normalize_string_server(prop_value)

    return feature.set(normalized_prop_name, normalized)


def prepare_rainfed_fields(
    landuse_collection: ee.FeatureCollection,
    double_cropping_image: ee.Image,
    not_irrigated_crops: Set[str],
    rainfed_crops: Set[str],
    minimum_field_size: int,
) -> ee.FeatureCollection:
    """
    Prepare rainfed fields by filtering and adding double cropping information.

    Args:
        landuse_collection (ee.FeatureCollection): Collection of land use features
        double_cropping_image (ee.Image): Image containing double cropping information
        not_irrigated_crops (List[str]): List of crop types that are not irrigated
        rainfed_crops (List[str]): List of rainfed reference crops
        minimum_field_size (int): Minimum field size in m^2

    Returns:
        ee.FeatureCollection: Filtered rainfed fields
    """
    landuse_collection = landuse_collection.map(normalize_feature)

    exclude_filter, rainfed_filter = create_crop_filters(
        not_irrigated_crops, rainfed_crops
    )

    nutzung_with_double_crop = add_double_cropping_info(
        landuse_collection, double_cropping_image
    )
    _, rainfed_fields = filter_crops(
        nutzung_with_double_crop, exclude_filter, rainfed_filter
    )

    # Add area property if not present
    rainfed_fields = rainfed_fields.map(
        lambda feature: feature.set("area", feature.geometry().area().divide(1).round())
    )

    # Drop all rainfed fields whose area is below minimum_field_size
    rainfed_fields = rainfed_fields.filter(ee.Filter.gte("area", minimum_field_size))

    return rainfed_fields


def prepare_fields(
    landuse_collection: ee.FeatureCollection,
    double_cropping_image: ee.Image,
    not_irrigated_crops: Set[str],
    rainfed_crops: Set[str],
    minimum_field_size: int,
) -> ee.FeatureCollection:
    """
    Prepare fields by filtering and adding double cropping information.

    Args:
        landuse_collection (ee.FeatureCollection): Collection of land use features
        double_cropping_image (ee.Image): Image containing double cropping information
        not_irrigated_crops (List[str]): List of crop types that are not irrigated
        rainfed_crops (List[str]): List of rainfed reference crops
        minimum_field_size (int): Minimum field size in m^2

    Returns:
        ee.FeatureCollection: Filtered rainfed fields
    """
    landuse_collection = landuse_collection.map(normalize_feature)

    exclude_filter, rainfed_filter = create_crop_filters(
        not_irrigated_crops, rainfed_crops
    )
    ## lets not use double cropping info for now
    # nutzung_with_double_crop = add_ALL_double_cropping_info(
    #     landuse_collection, double_cropping_image
    # )
    fields, rainfed_fields = filter_crops(
        landuse_collection, exclude_filter, rainfed_filter
    )

    # Add area property if not present
    fields = fields.map(
        lambda feature: feature.set("area", feature.geometry().area().divide(1).round())
    )

    # Drop all rainfed fields whose area is below minimum_field_size
    fields = fields.filter(ee.Filter.gte("area", minimum_field_size))

    return fields


def generate_export_task(
    et_green: ee.Image,
    asset_path: str,
    task_name: str,
    year: int,
    aoi: ee.Geometry,
    resolution: int = 10,
) -> ee.batch.Task:
    """
    Generate an export task for an ET green image.

    Args:
        et_green (ee.Image): ET green image to export
        asset_path (str): Base path for the asset
        task_name (str): Name of the export task
        year (int): Year being processed
        aoi (ee.Geometry): Area of interest
        resolution (int): Export resolution in meters

    Returns:
        ee.batch.Task: Export task
    """
    asset_id = f"{asset_path}/{task_name}"
    crs = et_green.projection().crs()

    task = export_image_to_asset(
        image=et_green,
        asset_id=asset_id,
        task_name=task_name,
        aoi=aoi,
        crs=crs,
        scale=resolution,
        year=year,
    )

    return task

def process_et_green(
    et_collection_list: ee.List,
    landuse_collection: ee.FeatureCollection,
    jurisdictions: ee.FeatureCollection,
    double_cropping_image: ee.Image,
    year: int,
    aoi: ee.Geometry,
    asset_path: str,
    et_band_name: str = "downscaled",
    time_step_type: str = "dekadal",
    resolution: int = 10,
    not_irrigated_crops: List[str] = None,
    rainfed_crops: List[str] = None,
    minimum_field_size=1000,
    export_band_name: str = "ET_green",
) -> None:
    """
    Process and export ET green images for a given year.

    Args:
        et_collection_list (ee.List): List of ET images
        landuse_collection (ee.FeatureCollection): Collection of land use features
        jurisdictions (ee.FeatureCollection): Collection of jurisdiction boundaries
        double_cropping_image (ee.Image): Double cropping classification image
        year (int): Year to process
        aoi (ee.Geometry): Area of interest
        asset_path (str): Base path for asset export
        et_band_name (str): Name of the ET band to process
        time_step_type (str): Type of time step ("dekadal" or "monthly")
        resolution (int): Export resolution in meters
        not_irrigated_crops (List[str]): List of crops to exclude
        rainfed_crops (List[str]): List of rainfed reference crops
        minimum_field_size (int): Minimum field size in m^2, defaults to 1000 (1 ha)
    """
    # Use default crop lists if none provided
    if not_irrigated_crops is None:
        not_irrigated_crops = get_crops_to_exclude()
    if rainfed_crops is None:
        rainfed_crops = get_rainfed_reference_crops()
    else:
        rainfed_crops = {normalize_string_client(crop) for crop in rainfed_crops}

    # Prepare rainfed fields
    rainfed_fields = prepare_rainfed_fields(
        landuse_collection,
        double_cropping_image,
        not_irrigated_crops,
        rainfed_crops,
        minimum_field_size,
    )

    tasks = []
    collection_size = ee.List(et_collection_list).size().getInfo()

    for i in range(collection_size):
        # Process ET image
        et_image = ee.Image(et_collection_list.get(i))

        # Get time step pattern from image date
        date = ee.Date(et_image.get("system:time_start"))
        # year = date.get("year").getInfo()
        time_step_pattern = get_time_step_pattern(date, time_step_type)

        et_green_2bands = compute_et_green_std(
            et_image, rainfed_fields, jurisdictions, et_band_name=et_band_name
        )
        # Convert to integer
        et_green = back_to_int(et_green_2bands.select('ET_green'), 100)
        et_green_std = back_to_int(et_green_2bands.select('ET_green_std'), 100)
        et_green_std = et_green_std.rename(f"{export_band_name}_std")
        et_green = et_green.addBands(et_green_std)

        # Create export task
        task_name = f"{export_band_name}_{time_step_type}_{year}_{time_step_pattern}"
        # task_name = f"ET_green_{time_step_type}_{year}_{time_step_pattern}"
        task = generate_export_task(
            et_green, asset_path, task_name, year, aoi, resolution
        )
        tasks.append(task)

    print(f"Generated {len(tasks)} export tasks for year {year}")

def process_et_green_RF(
    et_collection_list: ee.List,
    rainfed_collection: ee.FeatureCollection,
    year: int,
    aoi: ee.Geometry,
    asset_path: str,
    etf_collection_list: ee.List,
    forest_proximity: ee.Image,
    rhiresD: ee.ImageCollection,
    DEM: ee.Image,
    soil_properties: ee.Image,
    n_trees: ee.Image,
    et_band_name: str = "downscaled",
    time_step_type: str = "dekadal",
    resolution: int = 30,
    export_band_name: str = "ET_green",
    max_trees: int = 5,
    nutzung_filter_list: List[str] = None,
    vegetation_period_image: ee.Image = None,
    advanced_processing: bool = False,
    perimeter_of_interest: str = None,
    numPixels: int = 10000,
) -> None:
    """
    Apply a random forest classifier to model ET green for a given year.

    Args:
        et_collection_list (ee.List): List of ET images
        rainfed_collection (ee.FeatureCollection): Collection of RAINFED land use features
        year (int): Year to process
        aoi (ee.Geometry): Area of interest
        asset_path (str): Base path for asset export
        et_band_name (str): Name of the ET band to process
        time_step_type (str): Type of time step ("dekadal" or "monthly")
        resolution (int): Export resolution in meters
        etf_collection_list (ee.List): List of ETF images
        forest_proximity: proximity to forest in meters,
        DEM: Digital Elevation Model,
        soil_properties: Soil Suitability Map,
        n_trees: Image representing the numbers of trees per field,
        max_trees: Maximum number of trees per field
        nutzung_filter_list (List[str]): List of 'nutzung' values to filter the rainfed_collection
        vegetation_period_image (ee.Image): Vegetation period image with bands like firstStart, firstEnd, etc.
        advanced_processing (bool): Whether to apply advanced processing including:
            - Using vegetation period bands as classifier inputs
            - Filtering out extreme ET values (2nd and 98th percentiles)
            - Excluding suspected irrigation regions from training data
        perimeter_of_interest (str): Asset path to suspected irrigation regions to exclude from training
        numPixels (int): Number of pixels to sample for training, defaults to 10000
    """
    # Filter rainfed collection by nutzung values if provided
    if nutzung_filter_list is not None:
        rainfed_collection = rainfed_collection.filter(ee.Filter.inList('nutzung', nutzung_filter_list))
        print(f"Filtered rainfed collection by nutzung values: {nutzung_filter_list}")
    
    # Exclude suspected irrigation regions from training if advanced processing is enabled
    if advanced_processing and perimeter_of_interest is not None:
        perimeter_fc = ee.FeatureCollection(perimeter_of_interest)
        # Create a mask that excludes the perimeter of interest areas
        perimeter_mask = ee.Image().byte().paint(perimeter_fc, 0).unmask(1)
        print(f"Excluding suspected irrigation regions from training: {perimeter_of_interest}")
    else:
        perimeter_mask = None
    
    # Prepare layers:
    # Aspect (based on DEM)
    aspect = ee.Terrain.aspect(DEM)
    # Slope (based on DEM)
    slope = ee.Terrain.slope(DEM)
    # Elevation (based on DEM)
    elevation = DEM
    # Northing (calculated from the image's projection)
    northing = ee.Image.pixelLonLat().select('latitude')
    # Easting (calculated from the image's projection)
    easting = ee.Image.pixelLonLat().select('longitude')
    # Soil suitability: 
    soil_suitability = soil_properties#.select(['wsp_ord','wsp_UNK'])
    # Forest proximity:
    forest_proximity = forest_proximity

    #prepare predictor layer
    predictors = ee.Image.cat(
        aspect,
        slope,
        elevation,
        northing,
        easting,
        soil_suitability,
        forest_proximity,
    )
    
    # Add vegetation period bands if advanced processing is enabled
    if advanced_processing and vegetation_period_image is not None:
        # Select all vegetation period bands
        veg_period_bands = vegetation_period_image.select(['firstStart', 'secondEnd'])#, 'secondStart','firstEnd', 'isDoubleCropping'
        predictors = predictors.addBands(veg_period_bands)
        print("Added vegetation period bands to predictors")

    # Prepare Layers for Masking
    # Numbers of trees (less than max_trees)
    n_trees_mask = n_trees.lte(max_trees)

    rf_mask = ee.Image().byte().paint(rainfed_collection, 1).rename('rf_mask')

    tasks = []
    collection_size = ee.List(et_collection_list).size().getInfo()

    for i in range(collection_size):
        # Process ET image
        et_image = ee.Image(et_collection_list.get(i))
        etf_image = ee.Image(etf_collection_list.get(i))

        # Convert to integer
        et_image = back_to_int(et_image, 100)

        # Get time step pattern from image date
        date = ee.Date(et_image.get("system:time_start"))
        time_step_pattern = get_time_step_pattern(date, time_step_type)

        # filter the sum of rainfall over the last 10 days and add it to predictors
        rainfall_sum = rhiresD.filterDate(date.advance(-5, 'day'), date.advance(5, 'day')).sum()
        predictors = predictors.addBands(rainfall_sum.rename('rainfall_sum'))

        # ===== sampling & training (classification) =====
        target = et_image.rename('ET_green')
        # update with etf image mask and n_trees_mask, and rainfed fields mask
        target = target.updateMask(etf_image.mask()).updateMask(n_trees_mask).updateMask(rf_mask)
        
        # Apply perimeter exclusion mask if advanced processing is enabled
        if advanced_processing and perimeter_mask is not None:
            target = target.updateMask(perimeter_mask)
        
        #count the number of unmasked pixels
        count = target.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=aoi,
            scale=resolution,
            maxPixels=1e13
        ).get('ET_green')
        # print(f"Number of unmasked pixels in target ET image: {count.getInfo()}")
        stack = predictors.addBands(target)
        # grid = cover_by_grid(aoi, 0.1, 0.1)
        # print(f"Grid size: {grid.size().getInfo()} cells")

        ###SAMPLE STRATIFIED BY LARGE GRIDS!
        samples = stack.sample_by_grid(
            region=aoi,
            scale=resolution,
            numPixels=numPixels, #reduce if Computed value is too large. (Error code: 3)
            dx=0.1, dy=0.1,    # grid size (approx 10km at equator)
            tileScale=1
        )
        # count the number of samples
        sample_size = samples.size().getInfo()
        print(f"Sampled {sample_size} points for training")
        # Filter out extremes if advanced processing is enabled
        if advanced_processing:
            # Get crop-specific ET quantiles
            qs = samples.aggregate_array('ET_green').reduce(ee.Reducer.percentile([2, 98]))
            q2  = ee.Number(ee.Dictionary(qs).get('p2'))
            q98 = ee.Number(ee.Dictionary(qs).get('p98'))
            
            # Filter out extremes (trim)
            samples = samples.filter(ee.Filter.gte('ET_green', q2)) \
                            .filter(ee.Filter.lte('ET_green', q98))
        
        clf = ee.Classifier.smileRandomForest(numberOfTrees=100)\
            .train(features=samples, classProperty='ET_green',
                    inputProperties=predictors.bandNames())
        et_green = predictors.classify(clf).rename('ETgreen_class')

        ##get etgreen values at the sample points
        cell_points = et_green.sampleRegions(
            collection=samples,
            scale=resolution,
            tileScale=1,
            geometries=True
        )

        # Convert back to original scale (divide by 100 to get decimal values)
        et_green = et_green.divide(100).rename(export_band_name)

        # Create export task, if there are samples and if asset does not yet exist
        if sample_size > 0:
            if not ee.data.getInfo(f"{asset_path}/{export_band_name}_{time_step_type}_{year}_{time_step_pattern}"):
                # # print the first five sample points for inspection
                # print('Test ET_green sample:', cell_points.limit(5).aggregate_array('ETgreen_class').getInfo())
                task_name = f"{export_band_name}_{time_step_type}_{year}_{time_step_pattern}"
                task = generate_export_task(
                    et_green.clip(aoi).set('samples',samples.size()), asset_path, task_name, year, aoi, resolution
                )
                tasks.append(task)

    print(f"Generated {len(tasks)} export tasks for year {year}")


def sample_by_grid(self, region, scale, numPixels,
                   dx, dy, seed=42, geometries=False, tileScale=4):
    """
    Sample ~numPixels points across region, divided among grid cells
    proportional to cell area. Grid step dx, dy are in the CRS units of `region`.
    """
    # 1. build grid
    grid = cover_by_grid(region, dx, dy)
    # 2. add area + proportional quota
    grid = grid.map(lambda f: f.set("area_m2", f.geometry().area(1)))
    total_area = ee.Number(grid.aggregate_sum("area_m2"))

    def set_quota(f):
        frac = ee.Number(f.get("area_m2")).divide(total_area)
        n = frac.multiply(numPixels).round().toInt()
        return f.set("n", n)

    grid = grid.map(set_quota)

    # add a mask band to self - mask is 1 only if all bands are valid
    all_bands_valid = self.mask().reduce(ee.Reducer.allNonZero())
    self = self.addBands(ee.Image(1).updateMask(all_bands_valid).rename('mask_band'))

    # 3. sample each grid cell
    def sample_cell(i):
        f = ee.Feature(grid.toList(grid.size()).get(i))
        geom = f.geometry()
        n = ee.Number(f.get("n"))
        # fc = self.sample(region=geom, scale=scale, numPixels=n,
        #                  seed=seed, geometries=geometries, tileScale=tileScale)
        ## use stratified sampling with mask band to make sure get the the desired number of points
        fc = self.stratifiedSample(
            numPoints=n,
            classBand='mask_band',
            region=geom,
            scale=scale,
            seed=seed,
            tileScale=tileScale,
            geometries=True,
            dropNulls=True  # dropNulls to avoid issues with empty samples
        )
        return fc.map(lambda feat: feat.set("cell_id", f.get("id")))

    idx_list = ee.List.sequence(0, grid.size().subtract(1))
    return ee.FeatureCollection(idx_list.map(sample_cell)).flatten()

# Attach the method to ee.Image
ee.Image.sample_by_grid = sample_by_grid

def generate_grid(xmin, ymin, xmax, ymax, dx, dy):
    """Return a FeatureCollection of rectangles covering [xmin, ymin, xmax, ymax] with steps dx, dy."""
    xx = ee.List.sequence(xmin, xmax, dx)
    yy = ee.List.sequence(ymin, ymax, dy)

    cells = xx.map(
        lambda x: yy.map(
            lambda y: ee.Feature(
                ee.Algorithms.GeometryConstructors.Rectangle(
                    ee.List([x, y, ee.Number(x).add(dx), ee.Number(y).add(dy)])
                )
            )
        )
    ).flatten()

    return ee.FeatureCollection(cells)

def cover_by_grid(geom, dx, dy):
    """Cover 'geom' with a regular grid of cell size dx, dy (in the geometry's coordinate units)."""
    bounds = ee.Geometry(geom).bounds()
    coords = ee.List(bounds.coordinates().get(0))
    ll = ee.List(coords.get(0))
    ur = ee.List(coords.get(2))
    xmin = ll.get(0)
    ymin = ll.get(1)
    xmax = ur.get(0)
    ymax = ur.get(1)

    # Generate and clip to geom
    gridcells = generate_grid(xmin, ymin, xmax, ymax, dx, dy).filterBounds(geom)

    # Add an integer 'id' (1..N)
    size = gridcells.size()
    gridcells = ee.FeatureCollection(
        ee.List.sequence(0, size.subtract(1)).map(
            lambda i: ee.Feature(gridcells.toList(size).get(i)).set('id', ee.Number(i).add(1))
            # Add the x and y centroids
            .set('centroid_x', ee.Feature(gridcells.toList(size).get(i)).geometry().centroid().coordinates().get(0))
            .set('centroid_y', ee.Feature(gridcells.toList(size).get(i)).geometry().centroid().coordinates().get(1))
        )
    )


    # Intersect each cell with geom (buffer=100 to avoid topology issues, as in your JS)
    return gridcells.map(lambda f: ee.Feature(f).intersection(geom, 100))



def process_et_green_RF_uncertainty(
    et_collection_list: ee.List,
    rainfed_collection: ee.FeatureCollection,
    year: int,
    aoi: ee.Geometry,
    asset_path: str,
    etf_collection_list: ee.List,
    forest_proximity: ee.Image,
    rhiresD: ee.ImageCollection,
    DEM: ee.Image,
    soil_properties: ee.Image,
    n_trees: ee.Image,
    et_band_name: str = "downscaled",
    time_step_type: str = "dekadal",
    resolution: int = 30,
    export_band_name: str = "ET_green",
    max_trees: int = 5,
    nutzung_filter_list: List[str] = None,
    vegetation_period_image: ee.Image = None,
    advanced_processing: bool = False,
    perimeter_of_interest: str = None,
    numPixels: int = 20000,
) -> None:
    """
    Apply a random forest classifier to model ET green for a given year.

    Args:
        et_collection_list (ee.List): List of ET images
        rainfed_collection (ee.FeatureCollection): Collection of RAINFED land use features
        year (int): Year to process
        aoi (ee.Geometry): Area of interest
        asset_path (str): Base path for asset export
        et_band_name (str): Name of the ET band to process
        time_step_type (str): Type of time step ("dekadal" or "monthly")
        resolution (int): Export resolution in meters
        etf_collection_list (ee.List): List of ETF images
        forest_proximity: proximity to forest in meters,
        DEM: Digital Elevation Model,
        soil_properties: Soil Suitability Map,
        n_trees: Image representing the numbers of trees per field,
        max_trees: Maximum number of trees per field
        nutzung_filter_list (List[str]): List of 'nutzung' values to filter the rainfed_collection
        vegetation_period_image (ee.Image): Vegetation period image with bands like firstStart, firstEnd, etc.
        advanced_processing (bool): Whether to apply advanced processing including:
            - Using vegetation period bands as classifier inputs
            - Filtering out extreme ET values (2nd and 98th percentile)
            - Excluding suspected irrigation regions from training data
        perimeter_of_interest (str): Asset path to suspected irrigation regions to exclude from training
        numPixels (int): Number of pixels to sample for training, defaults to 20000
    """
    # Filter rainfed collection by nutzung values if provided
    if nutzung_filter_list is not None:
        rainfed_collection = rainfed_collection.filter(ee.Filter.inList('nutzung', nutzung_filter_list))
        print(f"Filtered rainfed collection by nutzung values: {nutzung_filter_list}")
    
    # Exclude suspected irrigation regions from training if advanced processing is enabled
    if advanced_processing and perimeter_of_interest is not None:
        perimeter_fc = ee.FeatureCollection(perimeter_of_interest)
        # Create a mask that excludes the perimeter of interest areas
        perimeter_mask = ee.Image().byte().paint(perimeter_fc, 0).unmask(1)
        print(f"Excluding suspected irrigation regions from training: {perimeter_of_interest}")
    else:
        perimeter_mask = None
    
    # Prepare layers:
    # Aspect (based on DEM)
    aspect = ee.Terrain.aspect(DEM)
    # Slope (based on DEM)
    slope = ee.Terrain.slope(DEM)
    # Elevation (based on DEM)
    elevation = DEM
    # Northing (calculated from the image's projection)
    northing = ee.Image.pixelLonLat().select('latitude')
    # Easting (calculated from the image's projection)
    easting = ee.Image.pixelLonLat().select('longitude')
    # Soil suitability: 
    soil_suitability = soil_properties#.select(['wsp_ord','wsp_UNK'])
    # Forest proximity:
    forest_proximity = forest_proximity

    #prepare predictor layer
    predictors = ee.Image.cat(
        aspect,
        slope,
        elevation,
        northing,
        easting,
        soil_suitability,
        forest_proximity,
    )
    residual_predictors = ee.Image.cat(
        # northing,
        # easting,
        aspect,
        slope,
        elevation,
        soil_suitability,
        forest_proximity
)
    # Add vegetation period bands if advanced processing is enabled
    if advanced_processing and vegetation_period_image is not None:
        # Select all vegetation period bands
        veg_period_bands = vegetation_period_image.select(['firstStart', 'secondEnd'])#, 'secondStart','firstEnd', 'isDoubleCropping'
        predictors = predictors.addBands(veg_period_bands)
        print("Added vegetation period bands to predictors")

    # Prepare Layers for Masking
    # Numbers of trees (less than max_trees)
    n_trees_mask = n_trees.lte(max_trees)

    rf_mask = ee.Image().byte().paint(rainfed_collection, 1).rename('rf_mask')

    tasks = []
    collection_size = ee.List(et_collection_list).size().getInfo()

    for i in range(collection_size):
        # Process ET image
        et_image = ee.Image(et_collection_list.get(i))
        etf_image = ee.Image(etf_collection_list.get(i))

        # Convert to integer
        et_image = back_to_int(et_image, 100)
        # # add random noise (pm 5) to et_image to avoid decimal steps in data, which the RF would replicate
        # et_image_tmp = et_image.add(ee.Image.random(42).multiply(10).subtract(5).round())
        # et_image = et_image_tmp.set("system:time_start", et_image.get("system:time_start"))

        # Get time step pattern from image date
        date = ee.Date(et_image.get("system:time_start"))
        time_step_pattern = get_time_step_pattern(date, time_step_type)

        # filter the sum of rainfall over the last 10 days and add it to predictors
        rainfall_sum = rhiresD.filterDate(date.advance(-5, 'day'), date.advance(5, 'day')).sum()
        predictors = predictors.addBands(rainfall_sum.rename('rainfall_sum'))

        # ===== sampling & training (classification) =====
        target = et_image.rename('ET_green')
        # update with etf image mask and n_trees_mask, and rainfed fields mask
        target = target.updateMask(etf_image.mask()).updateMask(n_trees_mask).updateMask(rf_mask)
        
        # Apply perimeter exclusion mask if advanced processing is enabled
        if advanced_processing and perimeter_mask is not None:
            target = target.updateMask(perimeter_mask)
        
        stack = predictors.addBands(target)
        ###SAMPLE STRATIFIED BY LARGE GRIDS!
        # samples = stack.sample(region=aoi, scale=resolution, numPixels=30000,
        #                     geometries=False, tileScale=4)
        samples = stack.sample_by_grid(
            region=aoi,
            scale=30,
            numPixels=numPixels,
            dx=0.1, dy=0.1,    # grid size (approx 10km at equator)
            tileScale=1
        )
        sample_size = samples.size().getInfo()
        print('Total Sample size:', sample_size)
    
        # Filter out extremes if advanced processing is enabled
        if advanced_processing:
            # Get crop-specific ET quantiles
            qs = samples.aggregate_array('ET_green').reduce(ee.Reducer.percentile([2, 98]))
            q2  = ee.Number(ee.Dictionary(qs).get('p2'))
            q98 = ee.Number(ee.Dictionary(qs).get('p98'))
            # print('ET_green 2nd percentile:', q2.getInfo())
            # print('ET_green 98th percentile:', q98.getInfo())
            # print('First 5 samples:', samples.limit(5).getInfo())
            # Filter out extremes (trim)
            samples = samples.filter(ee.Filter.gte('ET_green', q2)) \
                            .filter(ee.Filter.lte('ET_green', q98))
            
        #### 1) Train/test split for unbiased residuals
        samples = samples.randomColumn('rand', 42)
        train = samples.filter(ee.Filter.lt('rand', 0.5))
        test  = samples.filter(ee.Filter.gte('rand', 0.5))

        ############
        
        # 2) Train main ETgreen regressor on TRAIN
        clf = ee.Classifier.smileRandomForest(numberOfTrees=100)\
            .train(features=train, classProperty='ET_green',
                    inputProperties=predictors.bandNames())
        # et_green = predictors.classify(clf).rename('ETgreen_class')
        reg_main = clf

        # 3) Residuals on TEST points (obs - pred)
        # Apply classifier to FeatureCollection to get predictions
        test_pred = test.classify(reg_main, 'et_hat')
        test_res = test_pred.map(lambda f: f.set(
            'res', ee.Number(f.get('ET_green')).subtract(ee.Number(f.get('et_hat')))
        ))
        # Use absolute residual as target for the uncertainty model
        test_res = test_res.map(lambda f: f.set('abs_res', ee.Number(f.get('res')).abs()))


        # print('average residual:', test_res.aggregate_array('abs_res').reduce(ee.Reducer.mean()).getInfo())
        # print('median residual:', test_res.aggregate_array('abs_res').reduce(ee.Reducer.median()).getInfo())

        #         #Plot the x,y of the samples
        # # Get coordinates of all samples
        # def add_coordinates(feature):
        #     coords = feature.geometry().coordinates()
        #     return feature.set('x', coords.get(0), 'y', coords.get(1))
        
        # samples_with_coords = test_res.map(add_coordinates)
        
        # # Convert to Python lists for plotting
        # sample_data = samples_with_coords.select(['x', 'y', 'abs_res']).getInfo()
        
        # # Extract coordinates and abs_res values
        # x_coords = [feature['properties']['x'] for feature in sample_data['features']]
        # y_coords = [feature['properties']['y'] for feature in sample_data['features']]
        # et_values = [min(feature['properties']['abs_res'], 100) for feature in sample_data['features']]
        
        # # Create scatter plot
        # plt.figure(figsize=(12, 8))
        # scatter = plt.scatter(x_coords, y_coords, c=et_values, cmap='viridis', 
        #                     s=20, alpha=0.7, edgecolors='none', vmax=100)
        # plt.colorbar(scatter, label='abs_res (capped at 100)')
        # plt.xlabel('Longitude (x)')
        # plt.ylabel('Latitude (y)')
        # plt.title(f'Sample Locations and abs_res Values - Time Step {i+1}')
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        # plt.show()

        ###CALCULATE THE MEAN AND STANDARD DEVIATION OF THE RESIDUALS PER 10x10 GRID
                # 10 km grid example in CH (EPSG:2056)
        # Generate and clip to geom
        grid = cover_by_grid(aoi, 0.1, 0.1)
        if advanced_processing:
            #use a coarser grid if advanced processing to ensure not having regions with a lot of irrigation in one cel
            grid = cover_by_grid(aoi, 0.3, 0.3)


        # map over grid to calculate mean and standard deviation residual per cell
        def calc_cell_stats(i):
            f = ee.Feature(grid.toList(grid.size()).get(i))
            geom = f.geometry()
            cell_id = f.get('id')
            # filter test_res to points within this cell
            cell_points = test_res.filterBounds(geom)
            
            # Scale up before median calculations to preserve precision
            abs_res_list = cell_points.aggregate_array('abs_res')
            scaled_abs_res_list = abs_res_list#.map(lambda x: ee.Number(x).multiply(10000))
            
            # Calculate median and MAD using median reducer
            median_abs_res = ee.Number(scaled_abs_res_list.reduce(ee.Reducer.median()))
            
            mad_res = ee.Number(
                scaled_abs_res_list.map(lambda x: ee.Number(x).subtract(median_abs_res).abs())
                .reduce(ee.Reducer.median())
            )
            # mad_res = mad_scaled.divide(10000)
            mean_res = ee.Number(scaled_abs_res_list.reduce(ee.Reducer.mean()))
            stddev_res = ee.Number(scaled_abs_res_list.reduce(ee.Reducer.stdDev()))
            count = cell_points.size()
            return ee.Feature(geom, {
                'cell_id': cell_id,
                'mean_abs_res': mean_res,
                'mad_res': mad_res,
                'median_abs_res': median_abs_res,
                'stddev_abs_res': stddev_res,
                'count': count
            }).copyProperties(f, ['centroid_x', 'centroid_y'])
        cell_stats = ee.FeatureCollection(ee.List.sequence(0, grid.size().subtract(1)).map(calc_cell_stats))
        # print('Cell stats sample:', cell_stats.limit(5).getInfo())
        mean_abs_res = cell_stats.aggregate_array('mean_abs_res').reduce(ee.Reducer.mean())
        mad_res = cell_stats.aggregate_array('mad_res').reduce(ee.Reducer.median())
        median_abs_res = cell_stats.aggregate_array('median_abs_res').reduce(ee.Reducer.mean())
        stddev_abs_res = cell_stats.aggregate_array('stddev_abs_res').reduce(ee.Reducer.mean())

        # # Convert to Python lists for plotting
        # sample_data = cell_stats.select(['centroid_x', 'centroid_y', 'mean_abs_res']).getInfo()

        # # Extract coordinates and abs_res values
        # x_coords = [feature['properties']['centroid_x'] for feature in sample_data['features']]
        # y_coords = [feature['properties']['centroid_y'] for feature in sample_data['features']]
        # et_values = [feature['properties'].get('mean_abs_res', 0) for feature in sample_data['features'] if feature['properties'].get('mean_abs_res') is not None]
        
        # # Filter coordinates to match the filtered et_values
        # valid_indices = [i for i, feature in enumerate(sample_data['features']) if feature['properties'].get('mean_abs_res') is not None]
        # x_coords = [x_coords[i] for i in valid_indices]
        # y_coords = [y_coords[i] for i in valid_indices]

        # # Create scatter plot
        # plt.figure(figsize=(12, 8))
        # scatter = plt.scatter(x_coords, y_coords, c=et_values, cmap='viridis', 
        #         s=1000, alpha=0.7, edgecolors='none', vmax=100, marker='s')
        # plt.colorbar(scatter, label='mean_abs_res')
        # plt.xlabel('Longitude (x)')
        # plt.ylabel('Latitude (y)')
        # plt.title(f'Sample Locations and mean_abs_res Values - Time Step {i+1}')
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        # plt.show()

        # reduce to image for export: 1 km resolution
        # map over the list of properties to create an image with multiple bands
        listofprops = ['median_abs_res', 'mad_res', 'mean_abs_res', 'stddev_abs_res','count']
        def prop_to_image(prop):
            img = cell_stats.reduceToImage(
                properties=[prop],
                reducer=ee.Reducer.first()
            ).rename(prop)
            return img
        stats_image = ee.Image.cat(list(map(prop_to_image, listofprops)))
        # # Convert back to original scale (divide by 10000 to get decimal values), except count
        stats_image = stats_image.select(['median_abs_res', 'mad_res', 'mean_abs_res', 'stddev_abs_res'])\
                        .divide(100)\
                        .addBands(stats_image.select(['count']))

        #Mean of mean_abs_res across cells: 26.664827740263394
        #Mean of stddev_abs_res across cells: 24.230368940688546
        
        # # 4) Train a residual magnitude model on TEST points#
        # # Sample static predictors at the TEST points (properties already carried)
        # # (Because 'test' was sampled from stack, it has geometries; classification below reads properties)
        # res_err_reg = ee.Classifier.smileRandomForest(numberOfTrees=100)\
        #     .train(
        #         features=test_res,              # features must carry predictor properties
        #         classProperty='abs_res',
        #         inputProperties=residual_predictors.bandNames()
        #     )

        # # Apply to full AOI to get per-pixel expected absolute error (|residual|)
        # err_hat = residual_predictors.classify(res_err_reg).rename('abs_err_hat')  # same units as ET
        
        # # Get average predicted error with error handling
        # avg_err_result = err_hat.reduceRegion(
        #     reducer=ee.Reducer.mean(),
        #     geometry=aoi,
        #     scale=resolution,
        #     maxPixels=1e13
        # )
        # avg_err_hat = ee.Algorithms.If(
        #     avg_err_result.contains('abs_err_hat'),
        #     avg_err_result.get('abs_err_hat'),
        #     0
        # )
        # print('Average predicted absolute error (err_hat):', avg_err_hat.getInfo())

        # # Compare on the same mask used for ETgreen/ETblue
        # support = target.mask()  # or your crop fields mask
        # support_err_result = err_hat.updateMask(support).reduceRegion(
        #     reducer=ee.Reducer.mean(),
        #     geometry=aoi,
        #     scale=resolution,
        #     bestEffort=True, tileScale=4
        # )
        # mean_err_hat_on_support = ee.Algorithms.If(
        #     support_err_result.contains('abs_err_hat'),
        #     support_err_result.get('abs_err_hat'),
        #     0
        # )
        # print('avg err_hat (on support):', mean_err_hat_on_support.getInfo())

        # # 5) Calibrate a threshold multiplier k on the TEST set
        # # test_res has properties: 'res' and 'abs_res' on TEST points (with geometries)
        # # err_hat is an Image with band name 'abs_err_hat'

        # # Attach err_hat to the test points (adds property 'abs_err_hat')
        # test_with_e = err_hat.sampleRegions(
        #     collection=test_res,                # carry over existing properties
        #     properties=['res', 'abs_res'],      # keep residual fields
        #     scale=resolution,
        #     geometries=True,
        #     tileScale=1
        # )

        # # mean predicted error at TEST points
        # mean_err_hat_test = ee.Array(test_with_e.aggregate_array('abs_err_hat')) \
        #                         .reduce(ee.Reducer.mean(), [0]).get([0])
        # print('avg err_hat (at TEST points):', mean_err_hat_test.getInfo())
        # # print('Sample size for calibrating k:', test_with_e.size().getInfo())
        # # print('first 5 samples:', test_res.limit(1).getInfo())

        # # Ratio r = |res| / err_hat  (guard against zero and null values)
        # test_with_e = test_with_e.map(lambda f:
        #     f.set('r', ee.Algorithms.If(ee.Algorithms.IsEqual(f.get('abs_res'), None),None,                       
        #         ee.Algorithms.If(ee.Algorithms.IsEqual(f.get('abs_err_hat'), None),None,  # Set to None if either value is null
        #         ee.Number(f.get('abs_res')).divide(ee.Number(f.get('abs_err_hat')).max(1e-6))
        #     ))
        # ))
        # # print('Sample size for calibrating k:', test_with_e.size().getInfo())

        # # Calibrate k as the 95th percentile of r on held-out data
        # # Filter out null values before calculating percentile
        # test_with_e_valid = test_with_e.filter(ee.Filter.neq('r', None))
        # r_list = test_with_e_valid.aggregate_array('r')
        # percentile_result = r_list.reduce(ee.Reducer.percentile([95]))
        # k = ee.Algorithms.If(
        #     ee.Algorithms.IsEqual(percentile_result, None),
        #     1.0,  # Default value if no valid data
        #     ee.Number(percentile_result)  # dimensionless ratio
        # )
        # # print('Calibrated k (P95 of |res|/err_hat):', k.getInfo())

        # # Convert back to original scale (divide by 100 to get decimal values)
        # err_hat_with_k = err_hat.divide(100).rename(export_band_name)
        
        # # Convert absolute residuals from test points to image and add as extra band
        # abs_res_image = test_res.reduceToImage(
        #     properties=['abs_res'],
        #     reducer=ee.Reducer.mean()
        # ).rename('abs_residuals').divide(100)  # Convert to same scale

        # # Reproject using resample to match err_hat without calling projection()
        # # This avoids projection conflicts by letting GEE handle the reprojection automatically
        # abs_res_image = abs_res_image.reproject(
        #     crs='EPSG:4326',  # Use a standard projection
        #     scale=resolution  # Use the function parameter resolution
        # )

        # # Add absolute residuals as an extra band
        # # err_hat_with_k = abs_res_image#err_hat_with_k.addBands(abs_res_image)
        
        # # Save k95 as an image property AFTER all transformations
        # err_hat_with_k = err_hat_with_k.set({'k95':  k})
        
        # print('err_hat_with_k:', err_hat_with_k.getInfo())
        # Create export task
        if sample_size > 0:
            f = ee.Feature(grid.toList(grid.size()).get(10))
            geom = f.geometry()
            # filter test_res to points within this cell
            cell_points = test_res.filterBounds(geom)
            ### keeps generating values rounded to 10, due to scaling with 100. only alternative is adding random noise (pm 5), but this may result in Computed value is too large error!
            ## or alternatively do not apply 0.001 scaling during ET compositing, bcs this generates only a small range of integer values!
            # print('Test residuals sample:', cell_points.limit(5).aggregate_array('abs_res').getInfo())
            # print('Test ET_green sample:', cell_points.limit(5).aggregate_array('ET_green').getInfo())
            # print('Test et_hat sample:', cell_points.limit(5).aggregate_array('et_hat').getInfo())

            if not ee.data.getInfo(f"{asset_path}/{export_band_name}_{time_step_type}_{year}_{time_step_pattern}"):
                task_name = f"{export_band_name}_{time_step_type}_{year}_{time_step_pattern}"
                task = generate_export_task(
                    stats_image.clip(aoi).set('samples',test.size()), asset_path, task_name, year, aoi, resolution
                )
                tasks.append(task)

        # print('Mean of mean_abs_res across cells:', mean_abs_res.getInfo())
        # print('Median of mean_abs_res across cells:', median_abs_res.getInfo())
        # print('Mean of mad_res across cells:', mad_res.getInfo())
        # print('Mean of stddev_abs_res across cells:', stddev_abs_res.getInfo())

    print(f"Generated {len(tasks)} export tasks for year {year}")
