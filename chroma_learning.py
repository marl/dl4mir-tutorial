"""Train a multi-layer network in Theano to produce chroma features
from DFT Magnitude Coefficients.

Contact: <ejhumphrey@nyu.edu>
Homepage: http://marl.smusic.nyu.edu

This script will train a "deep" network to produce chroma. You need
a few things from the MARL website for it to run.
  1. Data
  2. Labels
  3. A map from chord strings to integers

Training will run for a predefined number of iterations, at which point the
parameters of the network will be saved to the specified pickle file. This
behavior can be realized at any time with the standard keyboard interrupt at
the command line (ctrl+C).

Sample call:
$ python chroma_learning.py \
chord_dft1025_train_data.npy \
chord_dft1025_train_labels.npy \
v157_chord_map.txt \
sample_params.pk \
--max_iterations 20000 \
--batch_size=100 \
--print_frequency 500 \
--learning_rate 0.02

Then you should see something like the following:
[Thu Nov 14 14:36:09 2013]   iter: 0000000  train loss: 3.0164
[Thu Nov 14 14:36:20 2013]   iter: 0000500  train loss: 2.3858
[Thu Nov 14 14:36:31 2013]   iter: 0001000  train loss: 2.3040
[Thu Nov 14 14:36:42 2013]   iter: 0001500  train loss: 1.7913
[Thu Nov 14 14:36:52 2013]   iter: 0002000  train loss: 1.8429
...
"""

import argparse
import cPickle
import json
import numpy as np
import theano
import theano.tensor as T
import time

from collections import OrderedDict

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


def generate_chroma_templates(num_qualities):
    """Generate chroma templates for some number of chord qualities.

    The supported qualities are, in order:
      [maj, min, maj7, min7, 7, maj6, min6, dim, aug, sus4, sus2, hdim7, dim7]

    Parameters
    ----------
    num_qualities : int
        Number of chord qualities to generate chroma templates.

    Returns
    -------
    templates : np.ndarray
        Array of chroma templates, ordered by quality. The first 12 are Major,
        the next 12 are minor, and so on.
    """
    templates = []
    position_idx = np.arange(12)
    # For all qualities ...
    for qual_idx in range(num_qualities):
        quality = QUALITIES[qual_idx]
        # Translate the string into a bit-vector.
        qual_array = np.array([int(v) for v in QUALITY_MAP[quality]])
        for root_idx in range(12):
            # Rotate for all roots, C, C#, D ...
            templates.append(qual_array[(position_idx - root_idx) % 12])

    templates.append(np.zeros(12))
    return np.array(templates)


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


def data_shuffler(data, labels, batch_size=100):
    """Data shuffler for training online algorithms with mini-batches.

    Parameters
    ----------
    data : np.ndarray
        Data observations with shape (n_samples, dim0, dim1, ... dimN).
    labels : np.ndarray
        Targets corresponding to the data (data).

    Yields
    ------
    x_m : np.ndarray
        Data with shape (batch_size, dim0, dim1, ... dimN).
    y_m : np.ndarray
        Targets corresponding to the samples in data_m.
    """
    num_samples = len(data)
    sample_idx = np.arange(num_samples, dtype=np.int32)
    read_ptr = num_samples

    while True:
        x_m, y_m = [], []
        while len(x_m) < batch_size:
            if read_ptr >= num_samples:
                np.random.shuffle(sample_idx)
                read_ptr = 0
            x_m.append(data[sample_idx[read_ptr]])
            y_m.append(labels[sample_idx[read_ptr]])
            read_ptr += 1
        yield np.array(x_m), np.array(y_m)


def prepare_training_data(train_file, label_file, label_map, batch_size=100):
    """Create a data generator from input data and label files.

    Parameters
    ----------
    train_file : str
        Path to a numpy file of data observations.
    label_file : str
        Path to a numpy file of data labels.
    label_map : dict
        Dictionary mapping string labels to integers.
    batch_size : int, default=100
        Number of datapoints to return for each batch.

    Returns
    -------
    shuffler : generator
        Data generator that returns an (x,y) tuple for each call
        to next().
    stats : dict
        Coefficient means and standard deviations, keyed by 'mu' and 'sigma'.
    """
    data, labels = np.load(train_file), np.load(label_file)
    y_true = np.array([label_map.get(l, -1) for l in labels])
    valid_idx = y_true > 0
    # Drop all labels that don't exist in the label map, i.e. negative.
    data, y_true = data[valid_idx], y_true[valid_idx]

    # Compute standardization statistics.
    stats = {'mu': data.mean(axis=0), 'sigma': data.std(axis=0)}

    num_qualities = int(y_true.max() / 12)
    templates = generate_chroma_templates(num_qualities)
    return data_shuffler(data, templates[y_true], batch_size=batch_size), stats


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


def build_network():
    """Build a chroma transform network for training.

    Returns
    -------
    objective_fx : compiled theano function
        Callable function that takes (x, y, eta) as arguments, returning the
        scalar loss over the data x; implicitly updates the parameters of the
        network given the learning rate eta.
    params : dict
        All trainable parameters in the network.
    """
    # ----------------------------------------------------
    # Step 1. Build the network
    # ----------------------------------------------------
    x_input = T.matrix('input')

    # Define layer shapes -- (n_in, n_out)
    l0_dim = (1025, 256)
    l1_dim = (256, 64)
    l2_dim = (64, 12)

    # Build-in the standardization methods.
    mu_obs = theano.shared(np.zeros(l0_dim[:1]), name='mu')
    sigma_obs = theano.shared(np.ones(l0_dim[:1]), name='sigma')
    x_input -= mu_obs.dimshuffle('x', 0)
    x_input /= sigma_obs.dimshuffle('x', 0)

    # Layer 0
    weights0 = theano.shared(np.random.normal(scale=0.01, size=l0_dim),
                             name='weights0')
    bias0 = theano.shared(np.zeros(l0_dim[1]), name='bias0')
    z_out0 = hwr(T.dot(x_input, weights0) + bias0)

    # Layer 1
    weights1 = theano.shared(np.random.normal(scale=0.01, size=l1_dim),
                             name='weights1')
    bias1 = theano.shared(np.zeros(l1_dim[1]), name='bias1')
    z_out1 = hwr(T.dot(z_out0, weights1) + bias1)

    # Layer 2
    weights2 = theano.shared(np.random.normal(scale=0.01, size=l2_dim),
                             name='weights2')
    bias2 = theano.shared(np.zeros(l2_dim[1]), name='bias2')
    z_output = hwr(T.dot(z_out1, weights2) + bias2)

    # ----------------------------------------------------
    # Step 2. Define a loss function
    # ----------------------------------------------------
    y_target = T.matrix('y_target')
    squared_distance = T.sum(T.pow(z_output - y_target, 2.0), axis=1)
    scalar_loss = T.mean(squared_distance)

    # ----------------------------------------------------
    # Step 3. Compute Update rules
    # ----------------------------------------------------
    eta = T.scalar(name="learning_rate")
    updates = OrderedDict()
    network_params = OrderedDict()
    for param in [weights0, bias0, weights1, bias1, weights2, bias2]:
        # Save each parameter for returning later.
        network_params[param.name] = param
        # Compute the gradient with respect to each parameter.
        gparam = T.grad(scalar_loss, param)
        # Now, save the update rule for each parameter.
        updates[param] = param - eta * gparam

    # ----------------------------------------------------
    # Step 4. Compile wicked fast theano functions!
    # ----------------------------------------------------
    # Function that computes the mini-batch loss *and* updates the network
    # parameters in-line.
    objective_fx = theano.function(inputs=[x_input, y_target, eta],
                                   outputs=scalar_loss,
                                   updates=updates,
                                   allow_input_downcast=True)

    # Add mu and sigma variables now, as we don't want to update them
    # during training.
    network_params.update({mu_obs.name: mu_obs,
                           sigma_obs.name: sigma_obs})
    return objective_fx, network_params


def train_network(objective_fx, shuffler, learning_rate, num_iterations,
                  print_frequency=100):
    """Run the training process for some number of iterations.

    Parameters
    ----------
    objective_fx : compiled theano function
        First function returned by build network; updates the parameters as
        data is passed to it.
    shuffler : generator
        Data source with a next() method, returning a two-element tuple (x,y).
    learning_rate : scalar
        Update rate for each gradient step.
    num_iterations : int
        Number of update iterations to run.
    print_frequency : int
        Number of iterations between printing information to the console.

    Returns
    -------
    train_loss : np.ndarray
        Vector of training loss values over iterations.
    """
    loss_values = np.zeros(num_iterations)
    n_iter = 0
    try:
        while n_iter < num_iterations:
            x_m, y_m = shuffler.next()
            loss_values[n_iter] = objective_fx(x_m, y_m, learning_rate)
            if (n_iter % print_frequency) == 0:
                print "[%s]\t iter: %07d \ttrain loss: %0.4f" % \
                    (time.asctime(), n_iter, loss_values[n_iter])
            n_iter += 1
    except KeyboardInterrupt:
        print "Stopping Early."

    return loss_values[:n_iter]


def save_parameters(params, output_file):
    """Collect all parameters in a dictionary and save to disk.

    Parameters
    ----------
    params : dict
        Symbolic Theano shared parameters keyed by name.
    output_file : string
        Path to output file.
    """
    param_values = dict()
    for name, param in params.iteritems():
        param_values[name] = param.get_value()

    file_handle = open(output_file, "w")
    cPickle.dump(param_values, file_handle)
    file_handle.close()


def main(args):
    """Main routine for training a deep network.

    Parameters
    ----------
    args : ArgumentParser
        Initialized argument object.
    """
    obj_fx, params = build_network()
    label_map = load_label_map(args.label_map)
    shuffler, stats = prepare_training_data(
        args.train_file, args.label_file, label_map, args.batch_size)

    # Set network's mu and sigma values.
    for name in ['mu', 'sigma']:
        params[name].set_value(stats[name])

    loss = train_network(obj_fx,
                         shuffler,
                         args.learning_rate,
                         args.max_iterations,
                         args.print_frequency)
    print "Final Loss: %s" % loss[-1]
    save_parameters(params, args.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Learn chroma features from DFT magnitude spectra.")
    parser.add_argument("train_file",
                        metavar="train_file", type=str,
                        help="Data for training.")
    parser.add_argument("label_file",
                        metavar="label_file", type=str,
                        help="Data labels for training.")
    parser.add_argument("label_map",
                        metavar="label_map", type=str,
                        help="JSON file mapping chord names to integers.")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Output file to save the model's parameters.")
    parser.add_argument("--max_iterations",
                        metavar="max_iterations", type=int,
                        default=5000, action="store",
                        help="Maximum number of iterations to train.")
    parser.add_argument("--batch_size",
                        metavar="batch_size", type=int,
                        default=50, action="store",
                        help="Size of the mini-batch.")
    parser.add_argument("--print_frequency",
                        metavar="print_frequency", type=int,
                        default=50, action="store",
                        help="Number of iterations between console printing.")
    parser.add_argument("--learning_rate",
                        metavar="learning_rate", type=float,
                        default=0.02, action="store",
                        help="Learning rate for updating parameters.")
    main(parser.parse_args())
