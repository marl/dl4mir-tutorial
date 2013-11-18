"""Transform a wavfile to chroma features and optionally save the output.

Contact: <ejhumphrey@nyu.edu>
Homepage: http://marl.smusic.nyu.edu

This mainfile demonstrates how to apply a trained network to never-before seen
wavefiles. Note that this is unlikely to be the most efficient strategy to
process a large collection of content, but it gets the point across.

Sample call:
$ python chroma_transform.py \
epiano-chords.wav \
epiano-chroma.npy \
sample_params.pk \
--hopsize=1024
"""

import argparse
import cPickle
import numpy as np
import theano
import theano.tensor as T
from scipy.io import wavfile


SAMPLERATE = 44100
FRAMESIZE = 2048


def signal_buffer(input_file, hopsize=441, batchsize=500):
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
        Matrix of sample data. The length of the final batch will almost
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
    while read_ptr < num_samples:
        idx0 = max([read_ptr - FRAMESIZE/2, 0])
        idx1 = min([read_ptr + FRAMESIZE/2, num_samples])
        x_m = waveform[idx0:idx1, 0]
        fidx = max([FRAMESIZE/2 - read_ptr, 0])
        frame[fidx:fidx+len(x_m)] = x_m
        batch.append(frame.copy())
        if len(batch) >= batchsize:
            yield np.asarray(batch)
            batch = list()
        read_ptr += hopsize
        frame[:] = 0
    yield np.asarray(batch)


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
    z_out0 = hwr(T.dot(x_input, weights0) + bias0)

    # Layer 1
    weights1 = theano.shared(param_values['weights1'], name='weights1')
    bias1 = theano.shared(param_values['bias1'], name='bias1')
    z_out1 = hwr(T.dot(z_out0, weights1) + bias1)

    # Layer 2
    weights2 = theano.shared(param_values['weights2'], name='weights2')
    bias2 = theano.shared(param_values['bias2'], name='bias2')
    z_output = hwr(T.dot(z_out1, weights2) + bias2)

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


def transformer(sigbuff, chroma_fx):
    """Main routine for iterating over a wavefile.

    Parameters
    ----------
    sigbuff : generator
        Initialized signal buffer.
    chroma_fx : compiled theano function
        Function that consumes 2D matrices of DFT coefficients and outputs
        chroma.

    Returns
    -------
    chroma : np.ndarray
        Matrix of time-aligned chroma vectors, shaped (num_frames, 12).
    """
    win = np.hanning(FRAMESIZE)[np.newaxis, :]
    z_output = [chroma_fx(np.abs(np.fft.rfft(win * x_m))) for x_m in sigbuff]
    return np.concatenate(z_output, axis=0)


def main(args):
    """Main routine for transforming a wavefile into chroma.

    Parameters
    ----------
    args : ArgumentParser
        Initialized argument object.
    """
    param_values = load_parameters(args.parameter_file)
    chroma_fx = build_network(param_values)
    sigbuff = signal_buffer(args.input_wavfile, hopsize=args.hopsize)
    chroma = transformer(sigbuff, chroma_fx)
    np.save(args.output_file, chroma)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Learn chroma features from DFT magnitude spectra.")
    parser.add_argument("input_wavfile",
                        metavar="input_wavfile", type=str,
                        help="Input file to transform.")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="File for saving chroma representation.")
    parser.add_argument("parameter_file",
                        metavar="parameter_file", type=str,
                        help="Parameter file for the chroma transformation.")
    parser.add_argument("--hopsize",
                        metavar="hopsize", type=int,
                        help="Hopsize for stepping through the waveform.")
    main(parser.parse_args())
