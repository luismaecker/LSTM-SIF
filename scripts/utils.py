# scripts/utils.py


from config import variables, min_time, max_time, lon_min, lon_max, lat_min, lat_max
from xcube.core.store import new_data_store
import rioxarray as rio
import os
import logging



# Function to setup logging to file
def configure_logging(filename):
    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

# Function to reset logging (so it creates a new file when run in the same session)
def reset_logging(filename):
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Remove all handlers associated with the root logger
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure logging with the new filename
    configure_logging(filename)


def create_dir(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))


def create_paths(data_path):

    # Create path and folder for germany border shapefile
    germany_shp_path = os.path.join(data_path, "germany_shape", 'germany_border.shp')
    create_dir(germany_shp_path)

    # Create path and folder for corine data year 2000
    corine_file_path = os.path.join(data_path, "landcover", f"forest_cover_2000.tif")
    create_dir(corine_file_path)

    # Create path and folderfor sif sample tif
    tif_sample_path= os.path.join(data_path, "cubes", "cube_sif_sample.tif")
    create_dir(tif_sample_path)

    cube_crop_path = os.path.join(data_path, "cubes", "cube_subset_crop.nc")

    return germany_shp_path, corine_file_path, tif_sample_path, cube_crop_path


def create_cube_subset(variables = variables, 
                        min_time = min_time, max_time = max_time,
                        lon_min = lon_min, lon_max = lon_max,
                        lat_min = lat_min, lat_max = lat_max):

    # Initalize xcube store
    store = new_data_store("s3", root="deep-esdl-public", storage_options=dict(anon=True))
    store.list_data_ids()

    # Load cube from store
    cube = store.open_data( 'esdc-8d-0.25deg-256x128x128-3.0.1.zarr')

    # Subset the cube
    cube_subset = cube.sel(time=slice(min_time, max_time)) \
                    .sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

    # select only specified variables from esdc
    cube_subset = cube_subset[variables]

    # Add crs to the cube
    cube_subset.rio.write_crs(4326, inplace = True)

    return cube_subset