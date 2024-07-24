# scripts/load_aux_data.py

import os
import rioxarray as rio
import matplotlib.pyplot as plt
import geopandas as gpd
import ee
import geemap
from utils import create_paths

# load custom function from utils.py


# Initialize GEE
def initialize_gee():
    ee.Authenticate(force=False)
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com', project='ee-forest-health')

# Download German border data
def download_german_border(path, download=False):

    print("Downloading German border data...")

    germany = ee.FeatureCollection('FAO/GAUL/2015/level0').filter(ee.Filter.eq('ADM0_NAME', 'Germany'))
       
    germany_geometry = germany.geometry()

    if download:
        geemap.ee_export_vector(germany, filename=path)

    print(100 * "-")

    return germany_geometry


# Download and preprocess Corine data
def load_corine(path, region, download=True):

    print("Processing Corine data...")

    landcover_collection = ee.ImageCollection('COPERNICUS/CORINE/V20/100m')

    landcover_year = landcover_collection.filterDate(f'1999-01-01', f'2000-12-31').first()

    zones = ee.Image(0) \
        .where(landcover_year.eq(311), 311) \
        .where(landcover_year.eq(312), 312) \
        .where(landcover_year.eq(313), 313)

    print("Downloading Corine data")

    if download:
        geemap.ee_export_image(zones, filename=path, crs="EPSG:4326", scale=500, region=region)

    print(100 * "-")


# Create sif sample tif for spatial resolution and transform
def create_sif_sample(out_path, cube_subset, write=True):

    cube_sample = cube_subset["sif_gosif"].isel(time=0)

    if write:
        cube_sample.rio.to_raster(out_path)

    print("Sample path created at:", out_path)

    print(100 * "-")



# Main workflow function
def load_aux_data(data_path, cube_subset, download = True):

    # Initialize GEE
    initialize_gee()

    # Create file paths and if they dont exist folders
    germany_shp_path, corine_file_path, tif_sample_path, _ = create_paths(data_path=data_path)

    # Download German border data 
    german_geometry = download_german_border(download=download, path=germany_shp_path)

    # Download and preprocess Corine data and use germany_geometry to define the AOI
    load_corine(path=corine_file_path, region=german_geometry, download=download)

    # Create sif sample tif
    create_sif_sample(out_path = tif_sample_path, cube_subset= cube_subset, write=download)



if __name__ == "__main__":

    print("Loading auxiliary data...")

    from utils import create_cube_subset
    
    data_path = "data"
    
    os.makedirs(data_path, exist_ok=True)

    # Create a subset of the Earth System Data Cube, containing only relevant variables and the desired spatial and temporal extent
    cube_subset = create_cube_subset()

    # Download auxiliary data (Germany border, Corine landcover data, sample tif)
    load_aux_data(data_path, cube_subset, download = True)

    print(100 * "-")

   