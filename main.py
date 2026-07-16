"""
Resample raw or epoched MEG/EEG data.

This app loads raw or epoched MNE data, resamples it to a target
sample rate (optionally jointly resampling an events file), and saves
the resampled data.

Inputs:
    - fif: Path to MNE raw or epoched .fif file
    - calibration, crosstalk, headshape, destination, events, channels: Optional companion files
    - param_epoched_data: Whether the input data is epoched (vs. continuous)
    - param_sfreq: New sample rate in Hz
    - param_npad: Padding amount for resampling
    - param_window: Frequency-domain window to use in resampling
    - param_stim_picks: Stim channels
    - param_n_jobs: Number of parallel jobs
    - param_raw_pad, param_epoch_pad: Padding type for raw/epoched data
    - param_save_jointly_resampled_events: Whether to save events resampled jointly with the data
    - param_pick_type: Channel type to keep before resampling

Outputs:
    - out_dir/raw.fif: Resampled data
    - out_dir/events.tsv: Jointly resampled events (if requested)
    - product.json: Metadata about the resampling
"""

# Copyright (c) 2026 brainlife.io
#
# Author: Franco Pestilli

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'brainlife_utils'))

# Standard imports
import mne
import warnings
import numpy as np
import shutil
import pandas as pd
from mne_bids import BIDSPath, write_raw_bids
from collections import Counter

# Import shared utilities
from brainlife_utils import (
    load_config,
    ensure_output_dirs,
    create_product_json,
    add_info_to_product,
    require_config_keys,
    read_optional_files,
    update_data_info_bads,
    define_kwargs
)

# Ensure output directories exist
ensure_output_dirs('out_dir')

# Load configuration
config = load_config()
require_config_keys(config, ['fif'])

product_items = []

# == LOAD DATA ==
data_file = config.pop('fif')
if config['param_epoched_data'] is False:
    data = mne.io.read_raw_fif(data_file, allow_maxshield=True)
else:
    data = mne.read_epochs(data_file)

# Read and save optional files
config, cross_talk_file, calibration_file, events_file, head_pos_file, channels_file, destination = read_optional_files(config, 'out_dir')

# Channels.tsv must be BIDS compliant
if channels_file is not None:
    user_warning_message_channels = f'The channels file provided must be ' \
                                    f'BIDS compliant and the column "status" must be present. '
    warnings.warn(user_warning_message_channels)
    add_info_to_product(product_items, user_warning_message_channels, 'warning')
    # Udpate data.info['bads'] with info contained in channels.tsv
    data, user_warning_message_channels = update_data_info_bads(data, channels_file)
    if user_warning_message_channels is not None:
        warnings.warn(user_warning_message_channels)
        add_info_to_product(product_items, user_warning_message_channels, 'warning')

# Extract the matrix of events #
if config['param_epoched_data'] is False:
    if events_file is not None:
        # Warning: events file must be BIDS compliant
        user_warning_message_events = f'The events file provided must be ' \
                                      f'BIDS compliant.'
        warnings.warn(user_warning_message_events)
        add_info_to_product(product_items, user_warning_message_events, 'warning')
        ############### TO BE TESTED ON NO RESTING STATE DATA
        # Compute the events matrix #
        df_events = pd.read_csv(events_file, sep='\t')

        # Extract relevant info from df_events
        samples = df_events['sample'].values
        event_id = df_events['value'].values

        # Compute the values for events matrix
        events_time_in_sample = [data.first_samp + sample for sample in samples]
        values_of_trigger_channels = [0]*len(events_time_in_sample)

        # Create a dataframe
        df_events_matrix = pd.DataFrame([events_time_in_sample, values_of_trigger_channels, event_id])
        df_events_matrix = df_events_matrix.transpose()

        # Convert dataframe to numpy array
        events_matrix = df_events_matrix.to_numpy()
    else:
        events_matrix = None
else:
    events_matrix = None

# Info message about resampling if applied
if config['param_epoched_data'] is False:
    add_info_to_product(product_items, f'Data was resampled at '
                                        f'{config["param_sfreq"]}. '
                                        f'Please bear in mind that it is generally '
                                        f'recommended not to epoch '
                                        f'downsampled data, but instead epoch '
                                        f'and then downsample.')

# Comment about resampling
comments_resample_freq = f'{config["param_sfreq"]}Hz'

# Check if the user will save an empty events file
if events_file is None and config['param_save_jointly_resampled_events'] is True:
    value_error_message = f'You cannot save en empty events file. ' \
                          f"If you haven't an events file, please set " \
                          f"'param_save_jointly_resampled_event' to False."
    # Raise exception
    raise ValueError(value_error_message)


## Convert parameters ##

# Deal with param_npad parameter #
# Convert param_npad into int if not "auto" when the App is run on BL
if config['param_npad'] != "auto":
    config['param_npad'] = int(config['param_npad'])

# Deal with param_n_jobs parameter #
# Convert n jobs into int when the App is run on BL
if config['param_n_jobs'] != 'cuda':
    config['param_n_jobs'] = int(config['param_n_jobs'])

# Deal with stim picks parameter #
# Convert stim picks into a list of int when the App is run on BL
if isinstance(config['param_stim_picks'], str) and config['param_stim_picks'] is not None:
    config['param_stim_picks'] = config['param_stim_picks'].replace('[', '')
    config['param_stim_picks'] = config['param_stim_picks'].replace(']', '')
    config['param_stim_picks'] = list(map(int, config['param_stim_picks'].split(', ')))

# Keep bad channels in memory
bad_channels = data.info['bads']

# Define the type of data
data = data.pick(picks=config['param_pick_type'])

# Delete keys values in config.json when this app is executed on Brainlife
del config['param_pick_type']
kwargs = define_kwargs(config)

# == RESAMPLE ==
# (parameters below come from kwargs, keyed identically to their config.json names)
data_copy = data.copy()

# For continuous data
if kwargs['param_epoched_data'] is False:

    # Load data
    data_copy.load_data()

    # Test if events file exist
    if events_matrix is not None and kwargs['param_save_jointly_resampled_events'] is True:

        # Resample data
        data_resampled, events_resampled = data_copy.resample(sfreq=kwargs['param_sfreq'], npad=kwargs['param_npad'], window=kwargs['param_window'],
                                                         stim_picks=kwargs['param_stim_picks'], n_jobs=kwargs['param_n_jobs'],
                                                         events=events_matrix, pad=kwargs['param_raw_pad'])
    else:
        # Resample data
        data_resampled = data_copy.resample(sfreq=kwargs['param_sfreq'], npad=kwargs['param_npad'], window=kwargs['param_window'],
                                       stim_picks=kwargs['param_stim_picks'], n_jobs=kwargs['param_n_jobs'],
                                       events=None, pad=kwargs['param_raw_pad'])
        events_resampled = None

# For epoched data
else:

    # Resample data
    data_resampled = data_copy.resample(sfreq=kwargs['param_sfreq'], npad=kwargs['param_npad'],
                                   window=kwargs['param_window'], n_jobs=kwargs['param_n_jobs'],
                                   pad=kwargs['param_epoch_pad'])
    events_resampled = None

# Save file
data_resampled.save(os.path.join('out_dir', 'raw.fif'), overwrite=True)

del data_copy

## Create BIDS compliant events file if existed ##
if events_resampled is not None and config['param_epoched_data'] is False:
    # Create a BIDSPath
    bids_path = BIDSPath(subject='subject',
                         session=None,
                         task='task',
                         run='01',
                         acquisition=None,
                         processing=None,
                         recording=None,
                         space=None,
                         suffix=None,
                         datatype='meg',
                         root='bids')

    # Extract event_id value #
    # to be tested when events are extracted from data
    event_id_value = list(events_resampled[:, 2])  # the third column of events corresponds to the value column of BIDS events.tsv
    id_values_occurrences = Counter(event_id_value)  # number of different events
    id_values_occurrences = list(id_values_occurrences.keys())
    trials_type = [f"events_{i}" for i in range(1, len(id_values_occurrences) + 1)]  # for trial type column of BIDS events.tsv
    dict_event_id = dict((k, v) for k, v in zip(trials_type, id_values_occurrences))

    # Write BIDS to create events.tsv BIDS compliant
    write_raw_bids(data, bids_path, events_data=events_resampled, event_id=dict_event_id, overwrite=True)

    # Extract events.tsv from bids path
    events_file = 'bids/sub-subject/meg/sub-subject_task-task_run-01_events.tsv'

    # Copy events.tsv in outdir
    shutil.copy2(events_file, os.path.join('out_dir', 'events.tsv'))

    # Info message in product.json
    add_info_to_product(product_items, 'Jointly resampled events are saved in events.tsv.')

# == CREATE PRODUCT.JSON ==
add_info_to_product(product_items, 'Data was successfully resampled.', 'success')
create_product_json(product_items)
