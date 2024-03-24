import mne
import numpy as np

# Preprocessing pipeline based on "The impact of artifact removal approaches on TMS–EEG signal" by Bertazzoli et al.
# Specifically based on the ARTIST pipeline

def remove_EOG(eeg_data):
    if("HEOG" in eeg_data.ch_names):
        eeg_data.drop_channels(["HEOG"])
    if("VEOG" in eeg_data.ch_names):
        eeg_data.drop_channels(["VEOG"])

def downsample(eeg_data, sample_rate = 1000):
    eeg_data.resample(sample_rate, npad="auto")

def get_baseline_corrected(eeg_data, events, event_dict, tmin=-1, tmax=-0.002):
    baseline = (tmin, tmax)

    # Create epochs
    epochs = mne.Epochs(eeg_data, events, event_dict, tmin, tmax, baseline=None, preload=True)

    # Apply baseline correction
    epochs.apply_baseline(baseline)
    evoked = epochs.average()

    # Convert evoked data to raw data
    raw_corrected = mne.io.RawArray(evoked.data, eeg_data.info)

    return raw_corrected

def get_baseline_corrected(eeg_data):
    data = eeg_data.get_data()
    channel_means = np.mean(data, axis=1)
    baseline_corrected_data = data - channel_means[:, np.newaxis]
    
    eeg_baseline_corrected = eeg_data.copy()
    eeg_baseline_corrected._data = baseline_corrected_data

    return eeg_baseline_corrected

def bandpass(eeg_data, low_freq = 1, high_freq = 90):
    eeg_data.filter(low_freq, high_freq)

def notch(eeg_data, freqs=[50]):
    eeg_data.notch_filter(freqs)

def calculate_range_indices(tms_index, start, end, sampling_rate):
    samples_before = int(start * sampling_rate)
    samples_after = int(end * sampling_rate)

    start_index = max(0, tms_index - samples_before)
    end_index = tms_index + samples_after

    return start_index, end_index

def remove_range(eeg_data_raw, tms_indices, start, end, sampling_rate, interpolate=True):
    """Replace all the data in the range around the TMS pulse with 0

    Args:
        eeg_data (RawBrainVision): Raw EEG data
        tms_indices (array): Timestamps of the TMS pulses
        start (int): Time before the TMS pulse to start removing data
        end (int): Time after the TMS pulse to stop removing data
        sampling_rate (int): Rate at which the EEG was sampled

    Returns:
        np.array: Filtered data
    """
    eeg_data = eeg_data_raw.get_data()
    num_electrodes = eeg_data.shape[0]
    if interpolate:
        for tms_index in tms_indices:
            start_index, end_index = calculate_range_indices(tms_index, start, end, sampling_rate)
            for i in range(num_electrodes):
                x = [start_index-1, end_index+1]
                y = [eeg_data[i, start_index-1], eeg_data[i, end_index+1]]
                x_new = np.arange(start_index, end_index+1)
                eeg_data[i, start_index:end_index+1] = np.interp(x_new, x, y)
            
    else:
        for tms_index in tms_indices:
            start_index, end_index = calculate_range_indices(tms_index, start, end, sampling_rate)
            for i in range(num_electrodes):
                eeg_data[i, start_index:end_index+1] = 0

    eeg_data_raw._data = eeg_data

def rereference(eeg_data):
    # mne.set_eeg_reference(eeg_data, ref_channels=['Cz'], projection=False)
    mne.set_eeg_reference(eeg_data, ref_channels='average')

def ICA_filter(eeg_data, events, event_dict, baseline=(-1, -0.002)):
    epochs = mne.Epochs(eeg_data, events, event_id=event_dict, tmin=-1, tmax=1, baseline=baseline, preload=True)
    ica = mne.preprocessing.ICA(n_components=20, random_state=42)
    ica.fit(epochs)
    muscle_idx_auto, scores = ica.find_bads_muscle(eeg_data)
    ica.exclude = muscle_idx_auto
    eeg_data = ica.apply(eeg_data)
    
def preprocess(eeg_data):
    """Filters the EEG data and removes artifacts. Changes the original data in place.

    Args:
        eeg_data (RawBrainVision): RawBrainVision object containing the EEG data

    Returns:
        np.array: numpy array of the preprocessed EEG data
    """
    sampling_rate = eeg_data.info['sfreq']

    events, event_dict = mne.events_from_annotations(eeg_data)
    tms_indices = [event[0] for event in events if event[2] == 1]
    
    remove_EOG(eeg_data) # done
    remove_range(eeg_data, tms_indices, 0.002, 0.005, sampling_rate) # done
    downsample(eeg_data) # done
    # detrending: fit a model and subtract model from signal
    ICA_filter(eeg_data, events, event_dict)
    bandpass(eeg_data)
    notch(eeg_data)
    ICA_filter(eeg_data, events, event_dict)
    rereference(eeg_data)
    # Replace the underneath with just baseline at epoching
    # get_baseline_corrected(eeg_data)
    return eeg_data