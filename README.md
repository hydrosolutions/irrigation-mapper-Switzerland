# Remote Sensing-Based Irrigation Mapping for Swiss Agriculture

A comprehensive irrigation mapping and agricultural water demand assessment framework developed for the Swiss Federal Office for the Environment (BAFU). This project combines satellite-based evapotranspiration data with land-use information to identify irrigated fields, quantify irrigation water consumption, and provide municipality- and canton-level irrigation statistics for Switzerland.

This repository contains the operational workflow developed for the BAFU study *«Satellitenfernerkundung zur Erfassung bewässerter Flächen und Bewässerungsmengen in der Schweiz - Validierung und methodische Weiterentwicklung am Beispiel des Bibertals im Kanton Schaffhausen»* (hydrosolutions GmbH, 2025). The workflow corresponds to the methods documented in the BAFU report and will be updated as the operational methodology evolves.

---

## Overview

The irrigation mapping workflow relies on actual evapotranspiration (ETa) products (Landsat Collection-2 L3 Provisional Actual ET; WaPOR V3) and compares them against rainfed reference ET (ETgreen) to estimate irrigation-driven water consumption (ETblue).

This methodology is consistent with the scientific approach documented in the accompanying BAFU report, where ET-based irrigation estimates were validated and applied in the Bibertal (Kanton Schaffhausen) and across the cantons Zürich, Thurgau and Schaffhausen.

Core objectives:

* Identify irrigated agricultural fields using ET-based indicators
* Estimate seasonal and annual irrigation volumes
* Provide irrigation statistics per municipality, district, and canton
* Enable reproducible, transparent, and scalable monitoring using Earth observation data
* Support water resources management under increasing climatic pressure

---

## Key Features

* Multi-sensor satellite integration (Landsat 7/8/9, Sentinel‑2)
* Machine-learning ETgreen modeling (Random Forest) using high‑resolution auxiliary datasets
* ETa–ETgreen comparison for ETblue estimation (irrigation water consumption)
* Integrated ETa/ETc stress filtering and noise detection
* Field‑level and municipality‑level irrigation statistics
* Time series analysis (2018–2024) with trend identification
* Fully reproducible workflow in Python + Google Earth Engine

## Methodology

The methodology follows the validated and extended framework developed in the 2025 BAFU report *«Satellitenfernerkundung zur Erfassung bewässerter Flächen und Bewässerungsmengen in der Schweiz»*. It integrates satellite-based ET products with machine learning to model ETgreen across heterogeneous agricultural landscapes.

### 1. Data Preparation

Preparation of ETa products (Landsat, WaPOR), land‑use polygons, and auxiliary datasets.

### 2. Vegetation Period Identification

NDVI-based extraction of crop-specific growing seasons using harmonised Sentinel‑2 time series.

### 3. ET Compositing

Creation of dekadal (10‑day) ETa composites for all available Landsat scenes.

### 4. Machine-Learning–Based ETgreen Modeling

A Random Forest model estimates **ETgreen** (rainfed evapotranspiration) using non‑irrigated reference surfaces (Weiden, Wiesen, Kunstwiesen) and high‑resolution auxiliary datasets:

* Copernicus DEM (Höhe, Hangneigung, Exposition)
* Bodenhinweiskarte (Sand/Schluff/Ton, SOC, CEC)
* MeteoSwiss RhiresD dekadische Niederschlagssummen
* NDVI-based phenology layers (Sentinel‑2)
* Spatial predictors (X/Y coordinates)
* Waldnähe (to reduce mixed‑pixel and TIR biases)

The model predicts ETgreen at 30 m resolution for every dekade across the three cantons.

### 5. ETblue Calculation

ETblue = ETa – ETgreen on a per‑pixel and per‑field basis.

Filtering steps:

* ETa/ETc thresholding to remove water‑stressed periods
* Residual-based noise detection
* Monthly ETblue thresholds for classification of irrigated fields

### 6. Visualization & Reporting

Field‑level and municipality‑level aggregation, mapping, statistics, and interactive HTML map visualizations.

## Core Algorithms

* ETblue = ETa − ETgreen
* Threshold-based irrigation detection (monthly ETblue minimum)
* ETa/ETc stress indicator filtering
* Temporal constraints (growing season: May–September)

---

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
└── utils/                         # Utility functions and helpers
```

---

## Data Sources

### Satellite Data

* **Landsat Collection‑2 L3 Provisional Actual ET** (USGS, SSEBop-based)
* **WaPOR V3.01** (ETLook-based, FAO)
* **Sentinel‑2 L2A** for NDVI time series

### Auxiliary Datasets (used in machine-learning ETgreen modeling)

* **Copernicus DEM GLO‑30** (height, slope, aspect)
* **Bodenhinweiskarte** (soil texture, SOC, CEC)
* **MeteoSwiss RhiresD** (daily precipitation, aggregated to dekades)
* **NDVI phenology layers** to capture crop timing
* **Bodenbedeckung Swisstopo** (forest)
* **Spatial predictors (X/Y coordinates)**
* **Swiss geodata** for land‑use polygons and boundaries

## Key Outputs

### Data Products

* Field-level irrigation maps (10–30 m)
* Municipality/district/canton-level irrigation statistics
* Annual and multi-year irrigation trends (2018–2024)
* Crop-group-specific irrigation water consumption

### Formats

* GeoTIFF (rasters, ET products)
* Shapefiles / GeoJSON (field geometries)
* CSV / Excel (statistics)

---

## Results & Applications

The methodology has been successfully applied in the BAFU study *«Satellitenfernerkundung zur Erfassung bewässerter Flächen und Bewässerungsmengen in der Schweiz»*, covering:

* The Bibertal (Kanton Schaffhausen) as the main validation and method‑development region
* Full‑extent application to the cantons Zürich, Thurgau and Schaffhausen (2018–2024)

These applications demonstrate the scalability and robustness of the machine‑learning–based ETgreen approach in Swiss agricultural landscapes.

### Use Cases

* Water resource allocation & planning
* Cantonal reporting and monitoring
* Climate adaptation strategies
* Agricultural extension services

---

## Citation

```
hydrosolutions GmbH (2025):
Satellitenfernerkundung zur Erfassung bewässerter Flächen und Bewässerungsmengen in der Schweiz -
Validierung und methodische Weiterentwicklung am Beispiel des Bibertals im Kanton Schaffhausen
Bericht zuhanden des Bundesamts für Umwelt (BAFU).
```

hydrosolutions GmbH (2025):
Abschätzung der bewässerten Fläche im Kanton Thurgau pro Gemeinde mit Fernerkundungsdaten.
Bericht zuhanden des Bundesamts für Umwelt (BAFU).

or

```
@misc{irrigation-mapper-bafu,
title={Remote Sensing-Based Irrigation Mapping for Swiss Agriculture},
author={Hydrosolutions GmbH},
year={2025},
url={https://github.com/hydrosolutions/irrigation-mapper-Switzerland}
}
```

---

## Contact

**Dr. Silvan Ragettli**\
Project Lead\
hydrosolutions GmbH\
Venusstrasse 29, 8050 Zürich\
Email: [ragettli@hydrosolutions.ch](mailto\:ragettli@hydrosolutions.ch)\
Phone: +41 43 535 05 80

```
