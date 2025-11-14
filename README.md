# Irrigation Mapper BAFU

A comprehensive irrigation mapping and water demand analysis framework for Switzerland, developed for the Swiss Federal Office for the Environment (BAFU). This project uses satellite-derived evapotranspiration data to identify irrigated agricultural fields and quantify irrigation water volumes at field and regional scales.

## Overview

This project implements a robust methodology for mapping irrigation patterns using Google Earth Engine, combining multiple satellite data sources (Landsat, Sentinel-2) with agricultural land use data to:

- **Identify irrigated fields** using evapotranspiration analysis
- **Quantify irrigation volumes** at field and regional scales  
- **Analyze temporal patterns** of irrigation across growing seasons
- **Generate comprehensive reports** for agricultural water management

## Key Features

- 🛰️ **Multi-sensor satellite data processing** (Landsat 8/9, Sentinel-2)
- 🌾 **Crop-specific irrigation analysis** (vegetables, maize, grasslands, sugar beets)
- 📊 **Field-level water demand quantification** 
- 🗺️ **Interactive visualization tools** and maps
- 📈 **Time-series analysis** (2018-2024) for trend assessment
- 🎯 **High-resolution mapping** (10-30m spatial resolution)

## Methodology

The project follows a systematic 6-step methodology:

1. **Data Preparation**: Processing evapotranspiration, land use, and rainfed reference field data
2. **Vegetation Period Extraction**: Identifying growing seasons and crop phenology
3. **ET Compositing**: Creating temporal composites of ET and ETf (ET fraction) data  
4. **ETgreen Modeling**: Computing rainfed evapotranspiration and residuals
5. **ETblue Calculation**: Quantifying irrigation water requirements per field
6. **Results Visualization**: Generating maps, statistics, and analytical reports

### Core Algorithms

- **ETblue = ETtotal - ETgreen**: Irrigation water = Total ET - Rainfed ET
- **Threshold-based classification**: ETa/ETc ratios and ET magnitude thresholds
- **Temporal filtering**: Growing season analysis (May-September)
- **Statistical validation**: Significance testing and outlier detection

## Repository Structure

```
irrigation-mapper-bafu/
├── src/                           # Core processing modules
│   ├── data_processing/           # Data preprocessing utilities
│   ├── et_blue/                   # ET blue computation algorithms  
│   ├── et_green/                  # ET green modeling
│   └── et_blue_per_field/         # Field-level processing
├── notebooks/                     # Analysis workflows (Jupyter notebooks)
│   ├── I_data_preparation/        # Step 1: Data preprocessing
│   ├── II_Vegetation_periods_extraction/  # Step 2: Phenology analysis
│   ├── III_decadal_compositing/   # Step 3: Temporal compositing
│   ├── IV_ETgreen_ETF_Residuals/  # Step 4: ETgreen modeling
│   ├── V_ETblue_per_field/        # Step 5: Field-level analysis
│   └── VI_results_visualization/  # Step 6: Results and visualization
├── utils/                         # Utility functions and helpers
├── vegetation_period_NDVI/        # NDVI-based vegetation period tools
├── data/                          # Processed data outputs
│   └── processed/                 # Analysis results and statistics
└── Figures/                       # Generated visualizations and maps
```

## Installation & Setup

### Prerequisites

- Python 3.8+ 
- Google Earth Engine account and authentication
- Required Python packages (see requirements below)

### Dependencies

```bash
# Core packages
pip install earthengine-api
pip install geemap
pip install pandas numpy matplotlib
pip install jupyter ipywidgets

# Earth Engine authentication
earthengine authenticate
```

### Google Earth Engine Setup

1. Create a Google Earth Engine account at [earthengine.google.com](https://earthengine.google.com)
2. Authenticate your environment:
   ```bash
   earthengine authenticate
   ```
3. Initialize in Python:
   ```python
   import ee
   ee.Initialize(project="your-project-name")
   ```

## Usage

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hydrosolutions/irrigation-mapper-bafu.git
   cd irrigation-mapper-bafu
   ```

2. **Run the complete workflow**:
   Navigate through the numbered notebook folders in sequence:
   ```
   notebooks/I_data_preparation/     → Data preprocessing
   notebooks/II_Vegetation_periods_extraction/  → Phenology extraction  
   notebooks/III_decadal_compositing/     → Temporal compositing
   notebooks/IV_ETgreen_ETF_Residuals/    → ETgreen modeling
   notebooks/V_ETblue_per_field/          → Field-level irrigation analysis
   notebooks/VI_results_visualization/    → Results visualization
   ```

3. **Field-level processing example**:
   ```python
   from src.et_blue_per_field.field_level_postprocessing import FieldLevelETProcessor
   
   # Initialize processor
   processor = FieldLevelETProcessor(config)
   
   # Process specific date and canton
   et_blue_image = processor.process_date("2023-07-15", ["Thurgau"])
   ```

### Configuration

Key parameters can be adjusted in the processing scripts:

```python
# Irrigation detection thresholds
ETa_ETc_threshold = 0.7      # Water stress threshold
ETblue_threshold = 3.0       # Minimum irrigation amount (mm)

# Spatial parameters  
resolution = 10              # Output resolution (meters)
minimum_field_size = 1000    # Minimum field size (m²)

# Temporal parameters
growing_season = [5, 9]      # May to September
```

## Data Sources

- **Satellite Imagery**: Landsat 8/9, Sentinel-2 (via Google Earth Engine)
- **Land Use Data**: Swiss Federal Statistical Office agricultural surveys
- **Meteorological Data**: Swiss meteorological stations
- **Administrative Boundaries**: Swiss cantons and municipalities

## Key Outputs

### Data Products
- **Irrigation Maps**: Field-level irrigation occurrence and intensity
- **Water Volume Statistics**: Monthly/seasonal irrigation volumes by region
- **Crop Analysis**: Irrigation patterns by crop type
- **Trend Analysis**: Multi-year irrigation trends (2018-2024)

### File Formats
- **Raster**: GeoTIFF (10-30m resolution)
- **Vector**: Shapefiles, GeoJSON
- **Tables**: CSV, Excel
- **Interactive**: HTML maps via geemap

## Results & Applications

### Key Findings
- Comprehensive irrigation mapping across Swiss agricultural regions
- Crop-specific irrigation water demand quantification
- Temporal analysis of irrigation patterns and trends
- Regional water use statistics for agricultural planning

### Use Cases
- **Water Resource Management**: Planning and allocation of irrigation water
- **Policy Development**: Evidence-based agricultural water policy
- **Climate Adaptation**: Understanding irrigation needs under climate change
- **Agricultural Planning**: Crop selection and irrigation system design

## Contributing

We welcome contributions to improve the irrigation mapping framework:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Create a Pull Request

### Development Guidelines
- Follow PEP 8 Python style guidelines
- Include docstrings for all functions
- Add unit tests for new functionality
- Update documentation for significant changes

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{irrigation-mapper-bafu,
  title={Irrigation Mapping Framework for Swiss Agriculture},
  author={Hydrosolutions and BAFU},
  year={2025},
  url={https://github.com/hydrosolutions/irrigation-mapper-bafu}
}
```

## License

This project is licensed under [LICENSE TYPE] - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Swiss Federal Office for the Environment (BAFU) for funding and support
- Google Earth Engine team for satellite data infrastructure
- Swiss Federal Statistical Office for agricultural land use data
- Contributors and collaborators in the Swiss water resources community

## Contact

- **Project Lead**: [Contact Information]
- **Technical Support**: [Email/Issues]
- **Organization**: Hydrosolutions
- **Website**: [Project Website]

## Version History

- **v1.0.0** (2025): Initial release with complete irrigation mapping workflow
- **v0.9.0** (2024): Beta version with field-level processing
- **v0.1.0** (2023): Initial development and proof of concept

---

*This project contributes to sustainable water resource management in Swiss agriculture through advanced satellite-based irrigation monitoring.*