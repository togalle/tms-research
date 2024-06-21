import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import mne_features
import concurrent.futures
from tqdm import tqdm
from scipy import stats
import utils

logger = utils.get_logger()

def feature_df(rseeg, sptep, rseeg_funcs, filename=None, n_jobs=2, save=True):
    filename = filename.split(".")[0] + ".csv"
    rseeg_df = mne_features.feature_extraction.extract_features(rseeg.get_data(copy=True), selected_funcs=rseeg_funcs, n_jobs=n_jobs, return_as_df=True, sfreq=rseeg.info["sfreq"], ch_names=rseeg.ch_names) # shape (num_epochs, num_features * num_channels)
    # sptep_df = compute_tep(sptep)
    # sptep_df = pd.concat([sptep_df]*len(rseeg_df), ignore_index=True)
    # feature_df = pd.concat([rseeg_df, sptep_df], axis=1)

    if save and filename is not None:
        rseeg_df.to_csv(filename, index=False)
    
    return rseeg_df


def load_fif_file(filename):
    eeg_data = mne.read_epochs(filename)
    return eeg_data


# Process file pairs
def process_file(filename_template, source_folder, selected_funcs, destination_folder=None):
    #filename template without file extension
    logger.info(f'Processing file pair {filename_template}')
    rseeg = load_fif_file(os.path.join(source_folder, filename_template.format(modality="rsEEG")))
    sptep = load_fif_file(os.path.join(source_folder, filename_template.format(modality="spTEP")))
    # destination_filename = filename_template.format(modality="").rstrip("_")
    df = feature_df(rseeg, sptep, selected_funcs, filename=os.path.join(destination_folder, filename_template.format(modality="rsEEG")), save=True)
    logger.info(df)


def feat_extr_on_folder(source_folder, destination_folder, parallel=True):
    # Make sure the destination folder for csv files exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get file pairs
    session_templates = set()
    pattern = re.compile(r'(spTEP|rsEEG)')
    files = os.listdir(source_folder)
    for filename in files:
        if pattern.search(filename):
            session_templates.add(pattern.sub('{modality}', filename))
    
    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_file, file, source_folder, destination_folder) for file in files]
                for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    pass
    else:
        for file in session_templates:
            process_file(file, source_folder, destination_folder)


def create_labels_csv(directory, metadata_csv, output_csv, modality_mapping={'rsEEG': 0, 'spTEP': 1}, timing_mapping={'pre': 0, 'post': 1}):
    # session mapping: {0: sham, 1: cTBS, 2: iTBS}
    metadata = pd.read_csv(metadata_csv, index_col=0, header=None)
    data = []

    for filename in os.listdir(directory):
        # note that the s can be upper or lower case and that the letter b can be behind the session number
        match = re.match(r'TMS-EEG-(-?)H_(\d+)_(S|s)(\w+)(b?)_(rsEEG|spTEP)_(pre|post)-epo.fif', filename)
        if match:
            _, patient_id, _, session, _, eeg_type, pre_post = match.groups()
            session = int(session.rstrip('b'))
            # procedure = procedure_labels[metadata.loc[f'H{patient_id}'][session]]
            procedure = metadata.loc[f'H{patient_id}'][session]
            patient_id = int(patient_id)
            eeg_type = modality_mapping[eeg_type]
            pre_post = timing_mapping[pre_post]
            filename = filename.split(".")[0]

            data.append([filename, procedure, patient_id, eeg_type, pre_post])
            
            logger.info(f'Added entry to labels.csv: {filename}, {patient_id}, {procedure}, {eeg_type}, {pre_post}')
    df = pd.DataFrame(data, columns=['filename', 'procedure', 'patient', 'modality', 'timing'])
    df.to_csv(output_csv, index=False, sep=",")