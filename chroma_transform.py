"""Transform a wavfile to chroma via DFT-folding and a learned transform.

Contact: <ejhumphrey@nyu.edu>
Homepage: http://marl.smusic.nyu.edu

This mainfile demonstrates how to apply a trained network to new data to
produce a chroma representation. This output is also compared to directly
folding the DFT magnitude spectra into a pitch-class profile (PCP).

Note: This script requires the input wavefile to have a samplerate of 11025Hz,
and will fail quite loudly in the event that the it does not.

Sample call:
$ python chroma_transform.py \
SMC_281.wav \
sample_params.pk \
--hopsize=1024
"""

import argparse
import cPickle
import numpy as np
import theano
import theano.tensor as T
from scipy.io import wavfile
from matplotlib.pyplot import figure, show

from dltutorial import chroma_tools as CT


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
    assert samplerate == CT.SAMPLERATE, \
        "Chroma transform only compatible with Fs==%d" % CT.SAMPLERATE
    num_samples = len(waveform)
    read_ptr = 0
    frame = np.zeros([CT.FRAMESIZE])
    batch = list()
    win = np.hanning(CT.FRAMESIZE)[np.newaxis, :]
    while read_ptr < num_samples:
        idx0 = max([read_ptr - CT.FRAMESIZE/2, 0])
        idx1 = min([read_ptr + CT.FRAMESIZE/2, num_samples])
        x_m = waveform[idx0:idx1]
        fidx = max([CT.FRAMESIZE/2 - read_ptr, 0])
        frame[fidx:fidx+len(x_m)] = x_m
        batch.append(frame.copy())
        if len(batch) >= batchsize:
            yield np.abs(np.fft.rfft(win * np.asarray(batch)))
            batch = list()
        read_ptr += hopsize
        frame[:] = 0
    yield np.abs(np.fft.rfft(win * np.asarray(batch)))


def build_network(param_values):
    """Build the one-layer chroma transform.

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


def audio_to_chroma(input_wavfile, hopsize, fx, norm=0):
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
    norm : scalar, default=0
        Lp norm to apply to the features; skipped if not > 0.

    Returns
    -------
    features : np.ndarray
        Matrix of time-aligned chroma vectors, shaped (num_frames, 12).
    """
    sigbuff = signal_buffer(input_wavfile, hopsize=hopsize)
    pitch_spec = np.concatenate([CT.cqt_pool(batch)
                                 for batch in sigbuff], axis=0)
    features = fx(pitch_spec)
    if norm > 0:
        features = CT.lp_norm(features, norm)
    return features


def mean_pitch_class(pitch_spec):
    """Compute average pitch class energy assuming octave equivalence.

    Parameters
    ----------
    pitch_spec : np.ndarray
        Array of pitch spectra.

    Returns
    -------
    chroma : np.ndarray
        Pitch-class features (chroma).
    """
    return np.array([pitch_spec[:, n::12].mean(axis=1) for n in range(12)]).T


def show_weights(param_file):
    """Plot the weights of the trained model.

    Parameters
    ----------
    param_file : str
        Path to the pickled file of network parameters.
    """
    params = cPickle.load(open(param_file))
    W = params['weights0']
    fig = figure()
    ax = fig.gca()
    ax.imshow(np.flipud(W.T), interpolation='nearest', aspect='auto')
    ax.set_yticks(range(12))
    ax.set_yticklabels(CT.PITCH_CLASSES[::-1])
    ax.set_ylabel("Pitch Class")
    c_ticks = range(len(W))[::12]
    ax.set_xticks(c_ticks)
    ax.set_xticklabels(["C%d" % (n + 1) for n in range(len(c_ticks))])
    ax.set_xlabel("Pitch")
    ax.tick_params(labelsize=10)
    show()


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
    learned_fx = build_network(param_values)

    dft_features = audio_to_chroma(
        args.input_wavfile, args.hopsize, mean_pitch_class, norm=1.0)

    learned_features = audio_to_chroma(
        args.input_wavfile, args.hopsize, learned_fx)

    fig = figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.imshow(
        np.flipud(dft_features.T), interpolation='nearest', aspect='auto')
    ax1.set_ylabel("DFT Chroma")
    ax1.set_yticks(range(12))
    ax1.set_yticklabels(CT.PITCH_CLASSES[::-1])
    ax1.tick_params(labelsize=10)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.imshow(
        np.flipud(learned_features.T), interpolation='nearest', aspect='auto')
    ax2.set_ylabel("Learned Chroma")
    ax2.set_yticks(range(12))
    ax2.set_yticklabels(CT.PITCH_CLASSES[::-1])
    ax2.tick_params(labelsize=10)
    ax2.set_xlabel("Frames")
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
