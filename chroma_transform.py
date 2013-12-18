"""Transform a wavfile to chroma features via DFT-folding and a deep network.

Contact: <ejhumphrey@nyu.edu>
Homepage: http://marl.smusic.nyu.edu

This mainfile demonstrates how to apply a trained network to new data to
produce a chroma representation. This output is also compared to directly
folding the DFT magnitude spectra into a pitch-class profile (PCP).

Note: This script requires the input wavefile to have a samplerate of 11025Hz,
and will fail quite loudly in the event that the it does not.

Sample call:
$ python chroma_transform.py \
epiano-chords.wav \
sample_params_20k.pk \
--hopsize=1024
"""

import argparse
import cPickle
import numpy as np
import theano
import theano.tensor as T
from scipy.io import wavfile
from matplotlib.pyplot import figure, show

SAMPLERATE = 11025
FRAMESIZE = 8192

PITCH_CLASSES = ['C', 'C#', 'D', 'Eb', 'E', 'F',
                 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def signal_buffer(input_file, hopsize=1024, batchsize=500):
    """Generator to step through an input wavefile.

    Note: The framesize is fixed due to preselected parameters.

    Parameters
    ----------
    input_file : str
        Path to an input wave file. Samplerate must be 44100 or the method
        will die loudly.
    hopsize : int
        Number of samples between frames.
    batchsize : int
        Number of frames to yield at a time.

    Yields
    -------
    batch : np.ndarray
        Array of DFT Spectra. The length of the final batch will almost
        certainly be smaller than the requested batchsize.
    """
    samplerate, waveform = wavfile.read(input_file)
    waveform = waveform.astype('float')*np.power(2.0, -15.0)
    assert samplerate == SAMPLERATE, \
        "Chroma transform only compatible with Fs==%d" % SAMPLERATE
    num_samples = len(waveform)
    read_ptr = 0
    frame = np.zeros([FRAMESIZE])
    batch = list()
    win = np.hanning(FRAMESIZE)[np.newaxis, :]
    while read_ptr < num_samples:
        idx0 = max([read_ptr - FRAMESIZE/2, 0])
        idx1 = min([read_ptr + FRAMESIZE/2, num_samples])
        x_m = waveform[idx0:idx1]
        fidx = max([FRAMESIZE/2 - read_ptr, 0])
        frame[fidx:fidx+len(x_m)] = x_m
        batch.append(frame.copy())
        if len(batch) >= batchsize:
            yield np.abs(np.fft.rfft(win * np.asarray(batch)))
            batch = list()
        read_ptr += hopsize
        frame[:] = 0
    yield np.abs(np.fft.rfft(win * np.asarray(batch)))


def hwr(x_input):
    """Theano functiom to compute half-wave rectification, i.e. max(x, 0).

    Parameters
    ----------
    x : theano symbolic type
        Object to half-wave rectify.

    Returns
    -------
    z : theano symbolic type
        Result of the function.
    """
    return 0.5 * (T.abs_(x_input) + x_input)


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


def build_network(param_values):
    """Build a chroma transform network for training.

    Parameters
    ----------
    param_values : dict
        Parameters for the network.

    Returns
    -------
    chroma_fx : compiled theano function
        Callable function that takes (x) as an argument; returns the chroma
        representation for the input data.
    """
    # ----------------------------------------------------
    # Step 1. Build the network
    # ----------------------------------------------------
    x_input = T.matrix('input')

    # Build-in the standardization methods.
    mu_obs = theano.shared(param_values['mu'], name='mu')
    sigma_obs = theano.shared(param_values['sigma'], name='sigma')
    x_input -= mu_obs.dimshuffle('x', 0)
    x_input /= sigma_obs.dimshuffle('x', 0)

    # Layer 0
    weights0 = theano.shared(param_values['weights0'], name='weights0')
    bias0 = theano.shared(param_values['bias0'], name='bias0')
    z_output = T.nnet.softmax(T.tanh(T.dot(x_input, weights0) + bias0))

    # ----------------------------------------------------
    # Step 2. Compile a wicked fast theano function!
    # ----------------------------------------------------
    chroma_fx = theano.function(inputs=[x_input],
                                outputs=z_output,
                                allow_input_downcast=True)
    return chroma_fx


def load_parameters(parameter_file):
    """Collect all parameters in a dictionary and save to disk.

    Parameters
    ----------
    parameter_file : str
        Path to a pickled file of parameters.

    Returns
    -------
    param_values : dict
        Numpy array parameter coefficients keyed by name.
    """
    return cPickle.load(open(parameter_file))


def audio_to_chroma(input_wavfile, hopsize, fx, norm=2.0):
    """Method for turning a wavefile into chroma features.

    Parameters
    ----------
    input_wavfile : str
        Path to a wavefile.
    hopsize : int
        Number of samples between frames.
    fx : function
        Function that consumes 2D matrices of DFT coefficients and outputs
        chroma features.
    norm : scalar
        Lp norm to apply to the features; skipped if not > 0.

    Returns
    -------
    features : np.ndarray
        Matrix of time-aligned chroma vectors, shaped (num_frames, 12).
    """
    sigbuff = signal_buffer(input_wavfile, hopsize=hopsize)
    features = np.concatenate([fx(batch) for batch in sigbuff], axis=0)
    if norm > 0:
        features = lp_norm(features, norm)
    return features


def cqt_pool(data):
    """write me, fool.
    """
    freqs = np.arange(FRAMESIZE/2 + 1, dtype=float)*SAMPLERATE/FRAMESIZE
    pitches = np.round(12*np.log2(freqs/440.0) + 69).astype(int)
    start_pitch = 24
    num_pitches = pitches.max() + 1 - start_pitch
    pitch_map = np.zeros([len(data), num_pitches])
    for bin_p in range(num_pitches):
        val = np.power(data[:, pitches == (bin_p + start_pitch)], 2.0).sum(axis=1) ** 0.5
        pitch_map[:, bin_p] += val
    return pitch_map


def dft_pcp(batch):
    """Baseline method for transforming DFT spectra into a pitch class profile.
    Derived from Fujishima (1999).

    Parameters
    ----------
    batch : np.ndarray
        Array of magnitude DFT spectra.

    Returns
    -------
    features : np.ndarray
        Pitch-class profile features for the batch.
    """
    freqs = np.arange(FRAMESIZE/2 + 1, dtype=float)*SAMPLERATE/FRAMESIZE
    pitches = np.round(12*np.log2(freqs/440.0) + 69).astype(int)
    num_pitches = pitches.max() + 1
    pitch_map = np.zeros([len(batch), num_pitches])
    for bin_p in range(24, num_pitches):
        val = np.power(batch[:, pitches == bin_p], 2.0).sum(axis=1) ** 0.5
        pitch_map[:, bin_p] += val

    return np.array([pitch_map[:, 24+n::12].mean(axis=1) for n in range(12)]).T


def main(args):
    """Main routine for transforming a wavefile into chroma by both the known,
    DFT-folding method and a learned transformation. The two representations
    are shown using matplotlib.

    Parameters
    ----------
    args : ArgumentParser
        Initialized argument object.
    """
    param_values = load_parameters(args.parameter_file)
    fx = build_network(param_values)

    def learned_fx(data):
        return fx(cqt_pool(data))

    learned_features = audio_to_chroma(
        args.input_wavfile, args.hopsize, learned_fx, 0.0)
    pcp_features = audio_to_chroma(
        args.input_wavfile, args.hopsize, dft_pcp, 2.0)

    fig = figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.imshow(np.flipud(pcp_features.T), interpolation='nearest', aspect='auto')
    ax2.imshow(np.flipud(learned_features.T), interpolation='nearest', aspect='auto')
    ax1.set_ylabel("PCP")
    ax1.set_yticks(range(12))
    ax1.set_yticklabels(PITCH_CLASSES[::-1])
    ax2.set_ylabel("Learned")
    ax2.set_yticks(range(12))
    ax2.set_yticklabels(PITCH_CLASSES[::-1])
    show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Learn chroma features from DFT magnitude spectra.")
    parser.add_argument("input_wavfile",
                        metavar="input_wavfile", type=str,
                        help="Input file to transform.")
    parser.add_argument("parameter_file",
                        metavar="parameter_file", type=str,
                        help="Parameter file for the chroma transformation.")
    parser.add_argument("--hopsize",
                        metavar="hopsize", type=int, default=1024,
                        help="Hopsize for stepping through the waveform.")
    main(parser.parse_args())
