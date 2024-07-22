# scripts/utils.py


from config import VARIABLES, min_time, max_time, lon_min, lon_max, lat_min, lat_max
from xcube.core.store import new_data_store
import rioxarray as rio

def create_cube_subset(VARIABLES = VARIABLES, 
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
    cube_subset = cube_subset[VARIABLES]

    # Add crs to the cube
    cube_subset.rio.write_crs(4326, inplace = True)

    return cube_subset