import xarray as xr
import rioxarray as rio
import numpy as np
import rasterio
from rasterio.windows import Window


# function to calculate forest percentages in a given window
def calculate_forest_percentage(lc_window, lc_data, forest_classes):

    forest_mask = np.isin(lc_data[lc_window.row_off:lc_window.row_off + lc_window.height,
                                  lc_window.col_off:lc_window.col_off + lc_window.width],
                          forest_classes)

    total_pixels = forest_mask.size
    forest_pixels = np.sum(forest_mask)
    percentage = (forest_pixels / total_pixels) * 100

    return percentage



def resample_corine_to_sif(corine_file_path, sample_path):
   
    # Open the landcover raster
    with rasterio.open(corine_file_path) as src_lc:
        lc_data = src_lc.read()
        lc_transform = src_lc.transform

    # Open the sample sif raster
    with rasterio.open(sample_path) as src_sif:
        sif_transform = src_sif.transform
        sif_meta = src_sif.meta

    # Resample the landcover raster to the sif raster
    
    # Determine the new shape and transform for the resampled raster
    new_height = sif_meta['height']
    new_width = sif_meta['width']

    # Initialize the new resampled data array
    resampled_forest_percentage = np.zeros((new_height, new_width), dtype=np.float32)

    # Define forest classes
    forest_classes = [311, 312, 313]

    # Calculate the window size in the original landcover data
    window_height = int(abs(sif_transform[4] / lc_transform[4]))
    window_width = int(abs(sif_transform[0] / lc_transform[0]))

    # Loop through each cell in the sif raster resolution
    for i in range(new_height):
        for j in range(new_width):
            # Define the window in the landcover data
            window = Window(col_off=j*window_width, row_off=i*window_height, width=window_width, height=window_height)
            
            # Calculate the forest percentage in the window
            forest_percentage = calculate_forest_percentage(window, lc_data.squeeze(), forest_classes)
            
            # Assign the percentage to the resampled data array
            resampled_forest_percentage[i, j] = forest_percentage

    resampled_forest_percentage_flip = np.flipud(resampled_forest_percentage)

        
    return resampled_forest_percentage_flip




def preprocess(cube_subset, germany_gpd, corine_file_path, sample_path, all_touched = True):

    # Clip the xarray dataset using the germany geometry
    cube_subset_crop = cube_subset.rio.clip(germany_gpd.geometry.values,
                                            germany_gpd.crs,
                                            drop = False, 
                                            all_touched = all_touched)
    
    resampled_forest_percentages = resample_corine_to_sif(corine_file_path, sample_path)

    # setup the dims for the resampled forest percentage
    dims = ('lat', 'lon')  

    # Add the resampled forest cover to the cube
    cube_subset_crop['forest_cover'] = xr.DataArray(resampled_forest_percentages, dims=dims, coords={dim: cube_subset_crop.coords[dim] for dim in dims})

    # Add a binary forest cover layer to the cube (0 for <50% forest cover, 1 for >=50% forest cover)
    cube_subset_crop['forest_cover_50'] = xr.DataArray((resampled_forest_percentages>=50).astype(int), dims=dims, coords={dim: cube_subset_crop.coords[dim] for dim in dims})

    return cube_subset_crop
