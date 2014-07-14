"""Chroma tools shared between learning and inference.
"""

from collections import OrderedDict
import json
import numpy as np

PITCH_CLASSES = ['C', 'C#', 'D', 'Eb', 'E', 'F',
                 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

QUALITIES = ['maj', 'min', 'maj7', 'min7', '7', 'maj6', 'min6',
             'dim', 'aug', 'sus4', 'sus2', 'hdim7', 'dim7']

QUALITY_MAP = {'maj':   '100010010000',
               'min':   '100100010000',
               'maj7':  '100010010001',
               'min7':  '100100010010',
               '7':     '100010010010',
               'maj6':  '100010010100',
               'min6':  '100100010100',
               'dim':   '100100100000',
               'aug':   '100010001000',
               'sus4':  '100001010000',
               'sus2':  '101000010000',
               'hdim7': '100100100010',
               'dim7':  '100100100100', }

SAMPLERATE = 11025
FRAMESIZE = 8192




def lp_norm(x, p):
    """Normalize the rows of x to unit norm in Lp-space.

    Parameters
    ----------
    x : np.ndarray
        Input matrix to normalize.
    p : scalar
        Shape of the metric space, e.g. 2=Euclidean.

    Returns
    -------
    z : np.ndarray
        Normalized representation.
    """
    s = np.power(np.power(np.abs(x), p).sum(axis=1), 1.0/p)
    s[s == 0] = 1.0
    return x / s[:, np.newaxis]


def load_label_map(filepath):
    """Load a human-readable (JSON) label map into memory.

    Note: JSON refuses to store integer zeros, so they are written as strings
    and must be interpreted as integers on load.

    Parameters
    ----------
    filepath : str
        Path to a JSON-ed text file mapping string labels to integers.

    Returns
    -------
    label_map : dict
        Mapping of string labels to integers
    """
    label_map = OrderedDict()
    for label, label_enum in json.load(open(filepath)).iteritems():
        label_map[label] = int(label_enum)
    return label_map


def cqt_pool(mag_spec):
    """Wrap DFT Magnitude spectra into a pitch map representation.

    Note: This function makes use of global constants to prevent mishaps.

    Parameters
    ----------
    mag_spec : np.ndarray
        Magnitude DFT coefficients, with shape (num_frames, NFFT/2 + 1).

    Returns
    -------
    pitch_spec : np.ndarray
        Pitch energy estimate.
    """
    freqs = np.arange(FRAMESIZE/2 + 1, dtype=float)*SAMPLERATE/FRAMESIZE
    pitches = np.round(12*np.log2(freqs/440.0) + 69).astype(int)
    num_pitches = pitches.max() + 1 - BIN0
    pitch_map = np.zeros([len(mag_spec), num_pitches])
    for bin_p in range(num_pitches):
        val_sqr = np.power(mag_spec[:, pitches == (bin_p + BIN0)], 2.0)
        pitch_map[:, bin_p] += np.power(val_sqr.sum(axis=1), 0.5)
    return pitch_map
