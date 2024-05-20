from rsEEG_cleaning import clean_rsEEG
from spTEP_cleaning import clean_spTEP
import mne
import os
import concurrent.futures
from tqdm import tqdm
from utils import get_logger

logger = get_logger()

def process_file(file_path, output_path):
    raw = mne.io.read_raw_brainvision(file_path, preload=True)

    if "spTEP" in file_path:
        epochs = clean_spTEP(file_path, raw, plot_result=True, save_result=False)
    elif "rsEEG" in file_path:
        epochs = clean_rsEEG(file_path, raw, plot_result=True, save_result=False)
    else:
        logger.error(f"Unknown file type: {file_path}")
        return
    
    output_file = os.path.join(output_path, os.path.basename(file_path))
    epochs.save(output_file)

def process_folder(folder_path, output_path):
    """
    Cleans all the files in the folder_path and saves the cleaned files in the output_path.
    """
    
    files = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith(".vhdr")
    ]
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file, output_path) for file in files]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass