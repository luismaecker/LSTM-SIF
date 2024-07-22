# scripts/load_aux_data.py

import os
import rioxarray as rio
import matplotlib.pyplot as plt
import geopandas as gpd
import ee
import geemap

# load custom function from utils.py


# Initialize GEE
def initialize_gee():
    ee.Authenticate(force=False)
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com', project='ee-forest-health')

# Download German border data
def download_german_border(path, download=False):

    germany = ee.FeatureCollection('FAO/GAUL/2015/level0').filter(ee.Filter.eq('ADM0_NAME', 'Germany'))
       
    germany_geometry = germany.geometry()
    if download:
        geemap.ee_export_vector(germany, filename=path)

    return germany_geometry


# Download and preprocess Corine data
def load_corine(landcover_collection,path, region, download=True):

    landcover_collection = ee.ImageCollection('COPERNICUS/CORINE/V20/100m')

    landcover_year = landcover_collection.filterDate(f'2000-01-01', f'2000-12-31').first()

    zones = ee.Image(0) \
        .where(landcover_year.eq(311), 311) \
        .where(landcover_year.eq(312), 312) \
        .where(landcover_year.eq(313), 313)

    if download:
        geemap.ee_export_image(zones, filename=path, crs="EPSG:4326", scale=500, region=region)

# Create sif sample tif for spatial resolution and transform
def create_sif_sample(sample_path, cube_subset, write=True):

    cube_sample = cube_subset["sif_gosif"].isel(time=0)

    if write:
        cube_sample.rio.to_raster(sample_path)

    return sample_path


# Main workflow function
def load_aux_data(data_path, cube_subset, download = True):

    # Initialize GEE
    initialize_gee()

    germany_shp_path = os.path.join(data_path, 'germany_border.shp')

    # Download German border data
    german_geometry = download_german_border(download=download, path=germany_shp_path)
    germany_gpd = gpd.read_file(germany_shp_path)

    # Create path for corine data year 2000
    corine_file_path = os.path.join(data_path, f"forest_cover_2000.tif")
    
    # Download and preprocess Corine data
    load_corine(path=corine_file_path, region=german_geometry, download=download)

    # Create path for sif sample tif
    tif_sample_path= os.path.join(data_path, "cube_sif_sample.tif")

    # Create sif sample tif
    sample_path = create_sif_sample(sample_path = tif_sample_path, cube_subset= cube_subset, write=download)

    return germany_gpd, corine_file_path, sample_path
