import xarray as xr
import rioxarray as rio
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from utils import create_paths, create_cube_subset

# Function to calculate forest percentages in a given window
def calculate_forest_percentage(lc_window, lc_data, forest_classes):
    """
    Calculate the percentage of forest cover in a specified window.

    Parameters:
    lc_window (Window): The window of the land cover data to analyze.
    lc_data (ndarray): The land cover data array.
    forest_classes (list): List of land cover classes considered as forest.

    Returns:
    float: Percentage of forest cover in the specified window.
    """
    forest_mask = np.isin(
        lc_data[lc_window.row_off:lc_window.row_off + lc_window.height,
                lc_window.col_off:lc_window.col_off + lc_window.width],
        forest_classes
    )

    total_pixels = forest_mask.size
    forest_pixels = np.sum(forest_mask)
    percentage = (forest_pixels / total_pixels) * 100

    return percentage

# Function to calculate the forest percentages of the CORINE land cover data over the cube grid
def resample_corine_to_sif(corine_file_path, sample_path):
    """
    Resample CORINE land cover data to match the resolution and dimensions of a sample SIF raster,
    and calculate forest cover percentages for each resampled cell.

    Parameters:
    corine_file_path (str): Path to the CORINE land cover file.
    sample_path (str): Path to the sample SIF raster file.

    Returns:
    ndarray: Array of forest cover percentages for each resampled cell.
    """
    # Open the land cover raster
    with rasterio.open(corine_file_path) as src_lc:
        lc_data = src_lc.read()
        lc_transform = src_lc.transform

    # Open the sample SIF raster
    with rasterio.open(sample_path) as src_sif:
        sif_transform = src_sif.transform
        sif_meta = src_sif.meta

    # Determine the new shape and transform for the resampled raster
    new_height = sif_meta['height']
    new_width = sif_meta['width']

    # Initialize the new resampled data array
    resampled_forest_percentage = np.zeros((new_height, new_width), dtype=np.float32)

    # Define forest classes
    forest_classes = [311, 312, 313]

    # Calculate the window size in the original land cover data
    window_height = int(abs(sif_transform[4] / lc_transform[4]))
    window_width = int(abs(sif_transform[0] / lc_transform[0]))

    # Loop through each cell in the SIF raster resolution
    for i in range(new_height):
        for j in range(new_width):
            # Define the window in the land cover data
            window = Window(col_off=j*window_width, row_off=i*window_height, width=window_width, height=window_height)
            
            # Calculate the forest percentage in the window
            forest_percentage = calculate_forest_percentage(window, lc_data.squeeze(), forest_classes)
            
            # Assign the percentage to the resampled data array
            resampled_forest_percentage[i, j] = forest_percentage

    resampled_forest_percentage_flip = np.flipud(resampled_forest_percentage)

    return resampled_forest_percentage_flip

def cube_preprocess(cube_subset, germany_gpd, corine_file_path, sample_path, out_path_crop, out_path_mask, all_touched=True, write=True):
    """
    Preprocess the data cube by clipping to Germany's border, calculating forest cover percentages,
    and adding this data to the cube. Optionally write the processed data to disk.

    Parameters:
    cube_subset (xarray.Dataset): The data cube subset to preprocess.
    germany_gpd (GeoDataFrame): GeoDataFrame containing Germany's borders.
    corine_file_path (str): Path to the CORINE land cover file.
    sample_path (str): Path to the sample SIF raster file.
    out_path_crop (str): Path to save the cropped data cube.
    out_path_mask (str): Path to save the masked data cube.
    all_touched (bool): Whether to include all pixels touched by the geometry. Defaults to True.
    write (bool): Whether to write the output to disk. Defaults to True.

    Returns:
    xarray.Dataset: The processed data cube subset.
    """
    print("Preprocessing cube")

    # Clip the xarray dataset using the Germany geometry
    print("Clipping cube to Germany border")
    cube_subset_crop = cube_subset.rio.clip(
        germany_gpd.geometry.values,
        germany_gpd.crs,
        drop=False, 
        all_touched=all_touched
    )
    
    # Calculate forest cover percentage over cube grid
    print("Calculate forest cover percentage over cube grid")
    resampled_forest_percentages = resample_corine_to_sif(corine_file_path, sample_path)

    # Setup the dimensions for the resampled forest percentage
    dims = ('lat', 'lon')  

    # Add the resampled forest cover to the cube
    cube_subset_crop['forest_cover'] = xr.DataArray(
        resampled_forest_percentages, dims=dims, coords={dim: cube_subset_crop.coords[dim] for dim in dims}
    )

    # Add a binary forest cover layer to the cube (0 for <50% forest cover, 1 for >=50% forest cover)
    cube_subset_crop['forest_cover_50'] = xr.DataArray(
        (resampled_forest_percentages >= 50).astype(int), dims=dims, coords={dim: cube_subset_crop.coords[dim] for dim in dims}
    )

    # Mask the cube where forest cover is less than 50%
    cube_subset_crop_mask = cube_subset_crop.where(cube_subset_crop['forest_cover_50'] == 1)

    if write:
        cube_subset_crop.to_netcdf(out_path_crop)
        cube_subset_crop_mask.to_netcdf(out_path_mask)
        print("Wrote cropped cube with added forest percentages and binary mask to disk at:", out_path_crop)
        print("Wrote cropped and masked cube to disk at:", out_path_mask)
                                       
    return cube_subset_crop


if __name__ == "__main__":
    from utils import create_cube_subset

    data_path = "data"

    # Load the cube subset
    cube_subset = create_cube_subset()

    # Create file paths and if they don't exist, create folders
    germany_shp_path, corine_file_path, tif_sample_path, cube_crop_path, cube_crop_mask_path = create_paths(data_path=data_path)

    # Load the Germany border geometry
    germany_gpd = gpd.read_file(germany_shp_path)

    # Preprocess the cube
    cube_preprocess(
        cube_subset, germany_gpd, corine_file_path, tif_sample_path, 
        out_path_crop=cube_crop_path, out_path_mask=cube_crop_mask_path, 
        all_touched=True, write=True
    )
