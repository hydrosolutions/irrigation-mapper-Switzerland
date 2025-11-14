"""
Field-level ET postprocessing for irrigation mapping.

This module provides a refactored Python implementation for processing ET, ETgreen,
and ETresidual images for any date within a given year. It replaces the hardcoded
JavaScript implementation with a flexible, configurable approach.

Author: Refactored from JavaScript original
Date: September 2025
"""

import ee
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import calendar

# Import existing utilities
from utils.ee_utils import back_to_float
from utils.date_utils import merge_same_date_images


@dataclass
class ProcessingConfig:
    """Configuration for ET processing."""
    year: int
    cantons: List[str] = None
    base_et_path: str = "projects/thurgau-irrigation/assets/{canton}/ET_products/decadal_Landsat_30m"
    etf_base_path: str = "projects/thurgau-irrigation/assets/{canton}/ET_products/decadal_Landsat_30m_ETF"
    etf_modeled_path: str = "projects/thurgau-irrigation/assets/ZH_SH_TG/ETF/ETF_Weiden_dekadal_from_Landsat_30m_v3"
    base_et_green_path: str = "projects/thurgau-irrigation/assets/ZH_SH_TG/ET_green"
    landuse_path: str = "projects/thurgau-irrigation/assets/ZH_SH_TG/Nutzungsflaechen/ZH_SH_TG_{year}_Kulturen"
    vegetation_period_path: str = "projects/thurgau-irrigation/assets/ZH_SH_TG/crop_vegetation_period_{year}_harmonic"
    wald_proximity_path: str = "projects/thurgau-irrigation/assets/Wald_SWISSTLM3D_2023_proximity"
    landuse_property_name: str = "nutzung"
    
    def __post_init__(self):
        if self.cantons is None:
            self.cantons = ["Schaffhausen","Thurgau", "Zuerich"]


@dataclass 
class CropTypeConfig:
    """Configuration for a specific crop type."""
    name: str
    landuse_categories: List[str]
    et_green_pattern: str
    et_residual_pattern: str


class FieldLevelETProcessor:
    """
    Process field-level ET calculations for irrigation mapping.
    
    This class provides methods to compute ET blue (irrigation water requirements)
    by processing ET, ET green, and ET residual images for any date within a year.
    """
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize the processor with configuration.
        
        Args:
            config: ProcessingConfig object containing paths and parameters
        """
        self.config = config
        self.crop_configs = self._setup_crop_configs()
        
        # Load static datasets
        self.landuse_collection = ee.FeatureCollection(
            self.config.landuse_path.format(year=self.config.year)
        )
        self.wald_proximity = ee.Image(self.config.wald_proximity_path)
        self.vegetation_period = ee.Image(
            self.config.vegetation_period_path.format(year=self.config.year)
        )
        self.double_cropping_mask = self.vegetation_period.select('isDoubleCropping').unmask(0)
        self.double_cropping_mask = self.double_cropping_mask.where(
            self.vegetation_period.select('secondStart').lte(5), 0
        )

        
    def _setup_crop_configs(self) -> Dict[str, CropTypeConfig]:
        """Setup configurations for different crop types."""
        return {
            "irrigated_vegetables": CropTypeConfig(
                name="irrigated_vegetables",
                landuse_categories=[
                    "Einjährige Freilandgemüse, ohne Konservengemüse",
                    "Mehrjährige gärtnerische Freilandkulturen (nicht im Gewächshaus)",
                    "Mehrjährige gärtnerische Freilandkulturen",
                    "Mehrjährige Gewürz- und Medizinalpflanzen",
                    "Kartoffeln",
                    "Freiland-Konservengemüse", 
                    "Freilandgemüse (ohne Kons.gemüse)",
                    "Soja",
                    "Spargel",
                    "Ölkürbisse",
                    "Oelkürbisse",
                    "Rhabarber",
                    "Pflanzkartoffeln (Vertragsanbau)",
                    "Einjährige Beeren (z.B. Erdbeeren)","Einjährige Beeren (Erdbeeren etc.)",
                    "Mehrjährige Beeren",
                    "Einjährige gärtnerische Freilandkulturen (Blumen, Rollrasen usw.)",
                    "Tabak",
                    "Einjä. gärtn. Freilandkult.(Blumen,Rollrasen)",
                    "Einjährige Freilandgemüse o. Konservengemüse"
                ],
                et_green_pattern="ET_green_Weiden_dekadal_from_Landsat_30m_v3/ET_green_dekadal_{year}_{month}_D{decad}",
                et_residual_pattern="ETg_residuals_Weiden_dekadal_from_Landsat_30m_v3/ETgreen_residuals_dekadal_{year}_{month}_D{decad}"
            ),
            "maize": CropTypeConfig(
                name="maize",
                landuse_categories=[
                    "Silo- und Grünmais",
                    "Körnermais",
                    "Saatmais (Vertragsanbau)"
                ],
                et_green_pattern="ET_green_Mais_dekadal_from_Landsat_30m_v3/ET_green_dekadal_{year}_{month}_D{decad}",
                et_residual_pattern="ETg_residuals_Mais_dekadal_from_Landsat_30m_v3/ETgreen_residuals_dekadal_{year}_{month}_D{decad}"
            ),
            "kunstwiese": CropTypeConfig(
                name="kunstwiese", 
                landuse_categories=[
                    "Kunstwiesen (ohne Weiden)",
                    "Kunstwiese (ohne Weiden)",
                    # "Übrige Dauerwiesen (ohne Weiden)" ##do not include them because some might have trees
                ],
                et_green_pattern="ET_green_Kunstwiesen_dekadal_from_Landsat_30m_v3/ET_green_dekadal_{year}_{month}_D{decad}",
                et_residual_pattern="ETg_residuals_Kunstwiesen_dekadal_from_Landsat_30m_v3/ETgreen_residuals_dekadal_{year}_{month}_D{decad}"
            ),
            "sugar_beet": CropTypeConfig(
                name="sugar_beet",
                landuse_categories=[
                    "Zuckerrüben"
                ],
                et_green_pattern="ET_green_Zuckerrueben_dekadal_from_Landsat_30m_v3/ET_green_dekadal_{year}_{month}_D{decad}",
                et_residual_pattern="ETg_residuals_Zuckerrueben_dekadal_from_Landsat_30m_v3/ETgreen_residuals_dekadal_{year}_{month}_D{decad}"
            )
        }
    
    def _get_date_components(self, date: Union[str, datetime]) -> Tuple[int, str, int]:
        """
        Extract year, month, and decadal components from a date.
        
        Args:
            date: Date as string (YYYY-MM-DD) or datetime object
            
        Returns:
            Tuple of (year, month_str, decad) where month_str is zero-padded
        """
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d")
        
        year = date.year
        month = f"{date.month:02d}"
        
        # Calculate decadal period (1-3 per month)
        day = date.day
        if day <= 10:
            decad = 1
        elif day <= 20:
            decad = 2
        else:
            decad = 3
            
        return year, month, decad
    
    def _construct_et_image_path(self, canton: str, date: Union[str, datetime]) -> str:
        """
        Construct ET image path for a specific canton and date.
        
        Args:
            canton: Canton name (Thurgau, Zuerich, Schaffhausen)
            date: Date for the image
            
        Returns:
            Full path to the ET image
        """
        year, month, decad = self._get_date_components(date)
        
        # Map canton names to path variations
        canton_mapping = {
            "Thurgau": "Thurgau", 
            "Zuerich": "Zurich",  # Note: path uses "Zurich" not "Zuerich"
            "Schaffhausen": "Schaffhausen"
        }
        
        path_canton = canton_mapping.get(canton, canton)
        base_path = self.config.base_et_path.format(canton=canton)
        image_name = f"ET_Landsat_decadal_{path_canton}_{year}{month}_{decad}"
        
        return f"{base_path}/{image_name}"
    
    def _construct_et_green_path(self, crop_config: CropTypeConfig, date: Union[str, datetime]) -> str:
        """
        Construct ET green image path for a crop type and date.
        
        Args:
            crop_config: Configuration for the crop type
            date: Date for the image
            
        Returns:
            Full path to the ET green image
        """
        year, month, decad = self._get_date_components(date)
        et_green_subpath = crop_config.et_green_pattern.format(
            year=year, month=month, decad=decad
        )
        return f"{self.config.base_et_green_path}/{et_green_subpath}"
    
    def _construct_et_residual_path(self, crop_config: CropTypeConfig, date: Union[str, datetime]) -> str:
        """
        Construct ET residual image path for a crop type and date.
        
        Args:
            crop_config: Configuration for the crop type  
            date: Date for the image
            
        Returns:
            Full path to the ET residual image
        """
        year, month, decad = self._get_date_components(date)
        et_residual_subpath = crop_config.et_residual_pattern.format(
            year=year, month=month, decad=decad
        )
        return f"{self.config.base_et_green_path}/{et_residual_subpath}"
    
    def load_et_image(self, date: Union[str, datetime]) -> Tuple[ee.Image, ee.Number]:
        """
        Load and mosaic ET images from all cantons for a specific date.
        
        Args:
            date: Date for the ET images
            
        Returns:
            Tuple of (mosaicked ET image, days scaling factor)
        """
        et_images = []
        
        for canton in self.config.cantons:
            image_path = self._construct_et_image_path(canton, date)
            et_images.append(ee.Image(image_path))
        
        # Create mosaic and copy properties from first image
        et_mosaic = ee.ImageCollection(ee.List(et_images)).mosaic()
        et_mosaic = et_mosaic.copyProperties(et_images[0])
        
        # Get days scaling factor
        days = ee.Number(et_mosaic.get('days'))
        
        return et_mosaic, days
    
    def _create_crop_mask(self, crop_config: CropTypeConfig) -> ee.Image:
        """
        Create a binary mask for a specific crop type.
        
        Args:
            crop_config: Configuration for the crop type
            
        Returns:
            Binary mask image (1 for crop fields, 0 elsewhere)
        """
        crop_collection = self.landuse_collection.filter(
            ee.Filter.inList(self.config.landuse_property_name, crop_config.landuse_categories)
        )
        
        return ee.Image().byte().paint(crop_collection, 1).unmask(0)
    
    def _process_crop_et_residual(self, crop_config: CropTypeConfig, date: Union[str, datetime]) -> ee.Image:
        """
        Process ET residual image for a crop type with proper masking.
        
        Args:
            crop_config: Configuration for the crop type
            date: Date for the images
            
        Returns:
            Processed ET residual image
        """
        et_residual_path = self._construct_et_residual_path(crop_config, date)
        et_residual = ee.Image(et_residual_path)
        
        # Create binary mask from the band's current mask
        bin_mask = et_residual.select('median_abs_res').mask().gt(0)
        
        # Apply the hardened mask
        return et_residual.mask(bin_mask)
    
    def process_crop_et_blue(self, 
                           crop_config: CropTypeConfig, 
                           et_image: ee.Image, 
                           days: ee.Number,
                           date: Union[str, datetime]) -> Tuple[ee.Image, ee.Image]:
        """
        Process ET blue calculation for a specific crop type.
        
        Args:
            crop_config: Configuration for the crop type
            et_image: Total ET image
            days: Days scaling factor
            date: Date for the processing
            
        Returns:
            Tuple of (ET blue image, ET residual image) for the crop
        """
        # Load ET green image
        et_green_path = self._construct_et_green_path(crop_config, date)
        et_green = ee.Image(et_green_path)
        
        # Load and process ET residual image
        et_residual = self._process_crop_et_residual(crop_config, date)
        
        # Create crop mask
        crop_mask = self._create_crop_mask(crop_config)
        
        # Calculate ET blue (irrigation requirement)
        et_blue = ee.Image(et_image).subtract(et_green.multiply(days))
        
        # Apply specific masking based on crop type
        if crop_config.name == "irrigated_vegetables":
            # For vegetables, apply forest proximity and double cropping masks
            et_blue = et_blue.updateMask(self.wald_proximity.gt(0))
            et_blue = et_blue.updateMask(
                self.double_cropping_mask.eq(1).Or(crop_mask.eq(1))
            )
            et_residual_scaled = et_residual.multiply(days).updateMask(
                self.double_cropping_mask.eq(1).Or(crop_mask.eq(1))
            )
        else:
            # For other crops, clip to crop fields
            et_blue = et_blue.clip(
                self.landuse_collection.filter(
                    ee.Filter.inList(self.config.landuse_property_name, crop_config.landuse_categories)
                )
            )
            et_residual_scaled = et_residual.multiply(days).clip(
                self.landuse_collection.filter(
                    ee.Filter.inList(self.config.landuse_property_name, crop_config.landuse_categories)
                )
            )
        
        return et_blue, et_residual_scaled
    
    def process_date(self, date: Union[str, datetime]) -> Tuple[ee.Image, ee.Image]:
        """
        Process all crop types for a specific date and create final blended images.
        
        Args:
            date: Date to process (YYYY-MM-DD string or datetime object)
            
        Returns:
            Tuple of (final ET blue image, final ET residual image)
        """
        # Load ET image and days factor
        et_image, days = self.load_et_image(date)
        
        # Process each crop type
        et_blue_images = []
        et_residual_images = []
        
        for crop_name, crop_config in self.crop_configs.items():
            et_blue, et_residual = self.process_crop_et_blue(
                crop_config, et_image, days, date
            )
            et_blue_images.append(et_blue)
            et_residual_images.append(et_residual)
        
        # Blend all crop-specific results
        final_et_blue = et_blue_images[0]
        for img in et_blue_images[1:]:
            final_et_blue = final_et_blue.blend(img)
        
        final_et_residual = et_residual_images[0] 
        for img in et_residual_images[1:]:
            final_et_residual = final_et_residual.blend(img)
            
        # Reproject residual to match ET blue projection
        final_et_residual = final_et_residual.reproject(
            final_et_blue.projection(), None, 30
        )
        
        return final_et_blue, final_et_residual
    
    def process_date4ETgreen(self, date: Union[str, datetime]) -> ee.Image:
        """
        Load and blend ET green images from all crop types for a specific date.
        
        Args:
            date: Date to process (YYYY-MM-DD string or datetime object)
            
        Returns:
            Blended ET green image from all crop types
        """
        # Get days scaling factor
        _, days = self.load_et_image(date)
        
        # Load ET green images for each crop type
        et_green_images = []
        
        for crop_name, crop_config in self.crop_configs.items():
            try:
                # Load ET green image for this crop type
                et_green_path = self._construct_et_green_path(crop_config, date)
                et_green = ee.Image(et_green_path)
                
                # Scale by days and create crop mask
                et_green_scaled = et_green.multiply(days)
                crop_mask = self._create_crop_mask(crop_config)
                
                # Apply crop-specific masking
                if crop_config.name == "irrigated_vegetables":
                    # For vegetables, apply forest proximity and double cropping masks
                    et_green_masked = et_green_scaled.updateMask(self.wald_proximity.gt(0))
                    et_green_masked = et_green_masked.updateMask(
                        self.double_cropping_mask.eq(1).Or(crop_mask.eq(1))
                    )
                else:
                    # For other crops, clip to crop fields
                    landuse_filter = self.landuse_collection.filter(
                        ee.Filter.inList(self.config.landuse_property_name, crop_config.landuse_categories)
                    )
                    et_green_masked = et_green_scaled.clip(landuse_filter)
                
                et_green_images.append(et_green_masked)
                
            except Exception as e:
                print(f"Warning: Failed to load ET green for {crop_config.name} on {date}: {e}")
                continue
        
        # Blend all crop-specific ET green images
        if not et_green_images:
            raise ValueError(f"No ET green images could be loaded for date {date}")
        
        final_et_green = et_green_images[0]
        for img in et_green_images[1:]:
            final_et_green = final_et_green.blend(img)
            
        return final_et_green
    
    def process_year(self, 
                    start_month: int = 5, 
                    end_month: int = 9,
                    time_step: str = "dekadal") -> List[Tuple[str, ee.Image, ee.Image]]:
        """
        Process all time steps for a growing season within a year.
        
        Args:
            start_month: Starting month (default: May = 5)
            end_month: Ending month (default: September = 9)
            time_step: Time step type ("dekadal" supported currently)
            
        Returns:
            List of tuples containing (date_string, et_blue_image, et_residual_image)
        """
        if time_step != "dekadal":
            raise ValueError("Currently only 'dekadal' time step is supported")
        
        results = []
        
        for month in range(start_month, end_month + 1):
            for decad in range(1, 4):
                # Calculate representative date for the decadal period
                if decad == 1:
                    day = 5  # Mid-point of 1st decad
                elif decad == 2:
                    day = 15  # Mid-point of 2nd decad  
                else:
                    # For 3rd decad, use a day that exists in all months
                    days_in_month = calendar.monthrange(self.config.year, month)[1]
                    day = min(25, days_in_month)  # Mid-point of 3rd decad, capped at month end
                
                date_str = f"{self.config.year}-{month:02d}-{day:02d}"
                
                try:
                    et_blue, et_residual = self.process_date(date_str)
                    results.append((date_str, et_blue, et_residual))
                except Exception as e:
                    print(f"Warning: Failed to process {date_str}: {e}")
                    continue
        
        return results


def main_processing_example():
    """
    Example usage of the FieldLevelETProcessor.
    """
    # Initialize Earth Engine
    ee.Initialize()
    
    # Setup configuration
    config = ProcessingConfig(year=2023)
    
    # Create processor
    processor = FieldLevelETProcessor(config)
    
    # Process a specific date (equivalent to the original JS code)
    date = "2023-08-25"  # 3rd decadal of August 2023
    et_blue, et_residual = processor.process_date(date)
    
    print(f"Processed ET blue and residual for {date}")
    print(f"ET blue bands: {et_blue.bandNames().getInfo()}")
    print(f"ET residual bands: {et_residual.bandNames().getInfo()}")
    
    # Process entire growing season
    seasonal_results = processor.process_year(start_month=5, end_month=9)
    print(f"Processed {len(seasonal_results)} time steps for the growing season")
    
    return processor, et_blue, et_residual, seasonal_results


if __name__ == "__main__":
    # Example usage
    processor, et_blue, et_residual, seasonal_results = main_processing_example()


def add_double_cropping_info(
    feature_collection: ee.FeatureCollection, veg_period_image: ee.Image, scale=10
) -> ee.FeatureCollection:
    """
    Adds double cropping information to each feature based on the median value of pixels within the feature.

    Args:
        feature_collection (ee.FeatureCollection): The input feature collection of crop fields.
        double_cropping_image (ee.Image): Image with 'isDoubleCropping' band (1 for double-cropped, 0 for single-cropped).
        scale (int): The scale to use for reducing the image.

    Returns:
        ee.FeatureCollection: Updated feature collection with 'isDoubleCropped' property.
    """
    


    # Create double cropping mask with the refined logic
    double_cropping_mask = veg_period_image.select('isDoubleCropping').unmask(0)
    double_cropping_mask = double_cropping_mask.where(
        veg_period_image.select('secondStart').lte(5), 0
    )

    # Update the vegetation period image to include the refined double cropping mask
    vegetation_period_updated = veg_period_image.addBands(
        double_cropping_mask, 
        overwrite=True
    )

    def add_double_crop_property(feature):
        median_value = (
            vegetation_period_updated
            .reduceRegion(
                reducer=ee.Reducer.median(),
                geometry=feature.geometry(),
                scale=scale,
            )
        )
        # # Create a new dictionary with rounded values
        # rounded_median_value = ee.Dictionary(
        #     median_value.map(lambda key, value: ee.Number(value).round())
        # )
        attrs = {
            "isDoubleCropping": ee.Algorithms.If(
                median_value.get("isDoubleCropping"), 
                ee.Number(median_value.get("isDoubleCropping")).round(), 
                0
            ),
            "firstStart": ee.Algorithms.If(
                median_value.get("firstStart"), 
                ee.Number(median_value.get("firstStart")).round(), 
                0
            ),
            "firstEnd": ee.Algorithms.If(
                median_value.get("firstEnd"), 
                ee.Number(median_value.get("firstEnd")).round(), 
                0
            ),
            "secondStart": ee.Algorithms.If(
                median_value.get("secondStart"), 
                ee.Number(median_value.get("secondStart")).round(), 
                0
            ),
            "secondEnd": ee.Algorithms.If(
                median_value.get("secondEnd"), 
                ee.Number(median_value.get("secondEnd")).round(), 
                0
            )
        }

        return feature.set(attrs)

    return feature_collection.map(add_double_crop_property)


# def assign_crop_class(landuse_value):
#     """
#     Assign crop class based on landuse value:
#     - vegetables and others: class 1
#     - maize: class 2  
#     - kunstwiesen: class 3
#     - zuckerrueben: class 4
#     - winter_crops: class 5
#     """
#     # Get crop type definitions from processor
#     vegetables_crops = set(processor.crop_configs["irrigated_vegetables"].landuse_categories)
#     maize_crops = set(processor.crop_configs["maize"].landuse_categories)
#     kunstwiese_crops = set(processor.crop_configs["kunstwiese"].landuse_categories)
#     sugar_beet_crops = set(processor.crop_configs["sugar_beet"].landuse_categories)
    
#     # Import winter crops
#     from src.et_green.filter_nutzungsflaechen import get_winter_crops
#     winter_crops = get_winter_crops()
    
#     if landuse_value in maize_crops:
#         return 2
#     elif landuse_value in kunstwiese_crops:
#         return 3
#     elif landuse_value in sugar_beet_crops:
#         return 4
#     elif landuse_value in winter_crops:
#         return 5
#     elif landuse_value in vegetables_crops:
#         return 1
#     else:
#         # All other crops default to class 1 (vegetables and others)
#         return 1

# Add the crop class as a server-side function for Earth Engine
def add_crop_class_to_fields(fields_collection):
    """Add crop class attribute to each field in the collection"""
    
    def add_class_property(feature):
        landuse = feature.get('nutzung')
        
        # Define the crop categories as Earth Engine lists
        vegetables_list = ee.List([
            "Einjährige Freilandgemüse, ohne Konservengemüse",
            "Mehrjährige gärtnerische Freilandkulturen (nicht im Gewächshaus)",
            "Mehrjährige gärtnerische Freilandkulturen",
            "Mehrjährige Gewürz- und Medizinalpflanzen",
            "Kartoffeln",
            "Freiland-Konservengemüse", 
            "Freilandgemüse (ohne Kons.gemüse)",
            "Soja",
            "Spargel",
            "Ölkürbisse",
            "Oelkürbisse",
            "Rhabarber",
            "Pflanzkartoffeln (Vertragsanbau)",
            "Einjährige Beeren (z.B. Erdbeeren)","Einjährige Beeren (Erdbeeren etc.)",
            "Mehrjährige Beeren",
            "Einjährige gärtnerische Freilandkulturen (Blumen, Rollrasen usw.)",
            "Tabak",
            "Einjä. gärtn. Freilandkult.(Blumen,Rollrasen)",
            "Einjährige Freilandgemüse o. Konservengemüse"
        ])
        
        maize_list = ee.List([
            "Silo- und Grünmais",
            "Körnermais",
            "Saatmais (Vertragsanbau)"
        ])
        
        kunstwiese_list = ee.List([
            "Kunstwiesen (ohne Weiden)",
            "Kunstwiese (ohne Weiden)"
        ])
        
        sugar_beet_list = ee.List([
            "Zuckerrüben"
        ])
        
        winter_crops_list = ee.List([
            "Wintergerste",
            "Winterweizen (ohne Futterweizen der Sortenliste swiss granum)",
            "Winterraps zur Speiseölgewinnung",
            "Dinkel",
            "Emmer, Einkorn",
            "Roggen",
            "Triticale",
            "Futterweizen gemäss Sortenliste swiss granum"
        ])
        
        # Assign class based on landuse category
        crop_class = ee.Algorithms.If(
            maize_list.contains(landuse), 2,
            ee.Algorithms.If(
                kunstwiese_list.contains(landuse), 3,
                ee.Algorithms.If(
                    sugar_beet_list.contains(landuse), 4,
                    ee.Algorithms.If(
                        winter_crops_list.contains(landuse), 5,
                        1  # Default to class 1 for vegetables and others
                    )
                )
            )
        )
        
        return feature.set('class', crop_class)
    
    return fields_collection.map(add_class_property)
