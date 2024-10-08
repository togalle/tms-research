import mne
import numpy as np

# Preprocessing pipeline based on "The impact of artifact removal approaches on TMS–EEG signal" by Bertazzoli et al.
# Specifically based on the ARTIST and TESA pipeline


def remove_EOG(eeg_data):
    if "HEOG" in eeg_data.ch_names:
        eeg_data.drop_channels(["HEOG"])
    if "VEOG" in eeg_data.ch_names:
        eeg_data.drop_channels(["VEOG"])


def remove_range(
    eeg_data_raw, tms_indices, start, end, sampling_rate, interpolate=True
):
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
            start_index, end_index = calculate_range_indices(
                tms_index, start, end, sampling_rate
            )
            for i in range(num_electrodes):
                x = [start_index - 1, end_index + 1]
                y = [eeg_data[i, start_index - 1], eeg_data[i, end_index + 1]]
                x_new = np.arange(start_index, end_index + 1)
                eeg_data[i, start_index : end_index + 1] = np.interp(x_new, x, y)

    else:
        for tms_index in tms_indices:
            start_index, end_index = calculate_range_indices(
                tms_index, start, end, sampling_rate
            )
            for i in range(num_electrodes):
                eeg_data[i, start_index : end_index + 1] = 0

    eeg_data_raw._data = eeg_data


def downsample(eeg_data, sample_rate=1000):
    eeg_data.resample(sample_rate, npad="auto")


def calculate_range_indices(tms_index, start, end, sampling_rate):
    samples_before = int(start * sampling_rate)
    samples_after = int(end * sampling_rate)

    start_index = max(0, tms_index - samples_before)
    end_index = tms_index + samples_after

    return start_index, end_index


def ICA_filter(eeg_data, events, event_dict, baseline=(-1, -0.002)):
    epochs = mne.Epochs(
        eeg_data,
        events,
        event_id=event_dict,
        tmin=-1,
        tmax=1,
        baseline=baseline,
        preload=True,
    )
    ica = mne.preprocessing.ICA(n_components=20, random_state=42)
    ica.fit(epochs)
    muscle_idx_auto, scores = ica.find_bads_muscle(eeg_data)
    ica.exclude = muscle_idx_auto
    eeg_data = ica.apply(eeg_data)


def bandpass(eeg_data, low_freq=1, high_freq=90):
    eeg_data.filter(low_freq, high_freq)


def notch(eeg_data, freqs=[50]):
    eeg_data.notch_filter(freqs)


def rereference(eeg_data):
    # mne.set_eeg_reference(eeg_data, ref_channels=['Cz'], projection=False)
    mne.set_eeg_reference(eeg_data, ref_channels="average")


def preprocess_spTEP(eeg_data):
    """Filters the EEG data and removes artifacts. Changes the original data in place.

    POSSIBLE IMPROVEMENT: demeaning instead of baseline correction

    Args:
        eeg_data (RawBrainVision): RawBrainVision object containing the EEG data

    Returns:
        np.array: numpy array of the preprocessed EEG data
    """
    sampling_rate = eeg_data.info["sfreq"]
    events, event_dict = mne.events_from_annotations(eeg_data)
    tms_indices = [event[0] for event in events if event[2] == 1]

    remove_EOG(eeg_data)
    remove_range(eeg_data, tms_indices, 0.002, 0.005, sampling_rate)
    downsample(eeg_data)
    ICA_filter(
        eeg_data, events, event_dict
    )  # TODO: apply ICA filter using a rule, not automatic detection
    bandpass(eeg_data)
    notch(eeg_data)
    ICA_filter(eeg_data, events, event_dict)
    rereference(eeg_data)
    return eeg_data


def preprocess_rsEEG(eeg_data):
    epoch_duration = 2
    overlap = 1
    sfreq = eeg_data.info["sfreq"]
    duration_samples = int(epoch_duration * sfreq)
    overlap_samples = int(overlap * sfreq)

    onset = np.arange(
        0,
        eeg_data.times[-1] * sfreq - duration_samples,
        duration_samples - overlap_samples,
    )

    events = np.vstack((onset, np.zeros_like(onset), np.ones_like(onset))).T.astype(int)
    event_id = 1

    remove_EOG(eeg_data)
    downsample(eeg_data)
    bandpass(eeg_data)
    notch(eeg_data)
    ICA_filter(eeg_data, events, event_id)
    rereference(eeg_data)
    return eeg_data
