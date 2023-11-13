"""
File: parc_labels.py
Author: Chuncheng Zhang
Date: 2023-11-13
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Amazing things

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2023-11-13 ------------------------
# Requirements and constants
from util import *

import numpy as np

import mne
from mne.datasets import eegbci
from mne.datasets import fetch_fsaverage

import matplotlib
import matplotlib.pyplot as plt

from rich import print, inspect

matplotlib.use('Qt5Agg')

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = Path(fs_dir).parent

# The files live in:
subject = "fsaverage"
trans = "fsaverage"  # MNE has a built-in fsaverage transformation
src = Path(fs_dir, "bem", "fsaverage-ico-5-src.fif")
bem = Path(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

print(mne.sys_info())

mne.viz.get_3d_backend()


# %% ---- 2023-11-13 ------------------------
# Function and class
channel_types_mapping = dict(
    HEO='eog',
    VEO='eog'
)

epochs_crop = dict(
    tmin=-0.2,
    tmax=1.0
)

filter_setup = dict(
    l_freq=0.1,
    h_freq=8.0,
    n_jobs=16
)

n_jobs = 8

# %% ---- 2023-11-13 ------------------------
# Play ground

cnts = [e for e in PPATH.data.iterdir() if e.name.endswith('.cnt')]
print(cnts)

cnt_file = cnts[3]
# cnt_file = cnts[0]
print(cnt_file)
raw = mne.io.read_raw_cnt(cnt_file)

try:
    raw.set_channel_types(channel_types_mapping)
except Exception:
    pass


def set_montage():
    '''
    Set the montage to the standard_1020,
    and convert the sensor's name into upper case.
    '''
    montage = mne.channels.make_standard_montage('standard_1020')

    names = montage.ch_names
    for n in names:
        montage.rename_channels({n: n.upper()})
    montage.rename_channels({'O9': 'CB1', 'O10': 'CB2'})

    raw.set_montage(montage, on_missing='warn')


set_montage()


# %%
# Extract events
events, event_id = mne.events_from_annotations(raw)
print(raw)
print(events)
print(event_id)

# %%
parc = 'HCPMMP1'
# parc = 'aparc'

labels_parc = mne.read_labels_from_annot(
    subject, parc=parc, subjects_dir=subjects_dir)
labels_parc_dict = {}
for e in labels_parc:
    assert e.name not in labels_parc_dict, 'Duplicated name'
    labels_parc_dict[e.name] = e

print(f'Found {len(labels_parc_dict)} labels')
# print('\n'.join(f'{e}, {len(v.values)}' for e, v in labels_parc_dict.items()))
print(labels_parc_dict)

# %%
label = labels_parc_dict['R_V1_ROI-rh']
label
inspect(label)

label.pos.shape
# %%
plot_bem_kwargs = dict(
    subject=subject,
    subjects_dir=subjects_dir,
    brain_surfaces="white",
    orientation="coronal",
    slices=[50, 100, 150, 200],
)

mne.viz.plot_bem(show=True, **plot_bem_kwargs)

mne.viz.plot_bem(src=src, show=True, **plot_bem_kwargs)

# %%
ss = mne.read_source_spaces(src)

# ss[1]['inuse'] *= 0
for i in label.vertices:
    ss[1]['inuse'][i] += 1

ss[1]['inuse'] -= 1
ss[1]['inuse'][ss[1]['inuse'] < 0] = 0

ss.plot()

# fig = ss.plot()
# inspect(ss, all=True)
print(ss)
print(ss[1])
inuse = ss[1]['inuse']
print(inuse.shape, np.sum(inuse))


# %%
fig = mne.viz.plot_alignment(
    subject=subject,
    subjects_dir=subjects_dir,
    surfaces="white",
    coord_frame="mri",
    src=src,
)

# %%
fig = mne.viz.plot_alignment(
    raw.info,
    src=src,
    eeg=["original", "projected"],
    trans=trans,
    show_axes=True,
    mri_fiducials=True,
    interaction='trackball',
    dig="fiducials",
)

input('>>')
# %% ---- 2023-11-13 ------------------------
# Pending


# %% ---- 2023-11-13 ------------------------
# Pending
