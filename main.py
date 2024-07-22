from scripts import load_aux_data
from utils import create_cube_subset
from preprocess import *









def main():
    
    data_path = "data"
    os.makedirs(data_path, exist_ok=True)

    # Create a subset of the Earth System Data Cube, containing only relevant variables and the desired spatial and temporal extent
    cube_subset = create_cube_subset()

    # Download auxiliary data (Germany border, Corine landcover data, sample tif)
    germany_gpd, corine_file_path, sample_path = load_aux_data(data_path, cube_subset, download = True)

    # Crop the cube to the extent of Germany and mask it with the Corine landcover data (50% forest cover)
    cube_subset_crop = preprocess(cube_subset, germany_gpd, corine_file_path, sample_path)


    base_analysis


    