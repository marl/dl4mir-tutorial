"""Train a multi-layer network in Theano to classify monophonic instruments.

Contact: <ejhumphrey@nyu.edu>

This script will train a "deep" network to classify monophonic instrument
feature vectors. You will need a few things from the MARL website for it to run.
  1. Data
  2. Labels

These will eventually be available at http://marl.smusic.nyu.edu

Sample call:
$ python instrument_demo.py \
instrument_dataset/uiowa_train_data.npy \
instrument_dataset/uiowa_train_labels.npy \
instrument_dataset/uiowa_test_data.npy \
instrument_dataset/uiowa_test_labels.npy \
--learning_rate=0.02 \
--print_frequency=100 \
--max_iterations=10000
"""

import argparse
import json
import numpy as np
import theano
import theano.tensor as T
import time

from collections import OrderedDict


def data_shuffler(data, labels, batch_size=100):
    """Data shuffler for training online algorithms with mini-batches.

    Parameters
    ----------
    data : np.ndarray
        Data observations with shape (n_samples, dim0, dim1, ... dimN).
    labels : np.ndarray
        Integer targets corresponding to the data (x).

    Returns
    -------
    x_m : np.ndarray
        Data with shape (batch_size, dim0, dim1, ... dimN).
    y_m : np.ndarray
        Targets corresponding to the samples in x_m.
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


def data_stepper(data, labels, batch_size=100):
    """Generator for stepping through a dataset in batches.

    Parameters
    ----------
    data : np.ndarray
        Data observations with shape (n_samples, dim0, dim1, ... dimN).
    labels : np.ndarray
        Integer targets corresponding to the data (x).

    Returns
    -------
    x_m : np.ndarray
        Data with shape (batch_size, dim0, dim1, ... dimN).
    y_m : np.ndarray
        Targets corresponding to the samples in x_m.
    """
    num_samples = len(data)
    read_ptr = 0
    x_m, y_m = [], []
    while read_ptr < num_samples:
        x_m.append(data[read_ptr])
        y_m.append(labels[read_ptr])
        read_ptr += 1
        if len(x_m) == batch_size:
            yield np.array(x_m), np.array(y_m)
            x_m, y_m = [], []

    if len(x_m) > 0:
        yield np.array(x_m), np.array(y_m)


def prepare_training_data(data_file, label_file, batch_size=100):
    """Create a data generator from input data and label files.

    Parameters
    ----------
    data_file : str
        Path to a numpy file of data observations.
    label_file : str
        Path to a numpy file of data labels.
    batch_size : int, default=100
        Number of datapoints to return for each batch.

    Returns
    -------
    shuffler : generator
        Data generator that returns an (x,y) tuple for each call
        to next().
    """
    data = np.load(data_file)
    labels = np.load(label_file)
    # Compute statistics for standardizing the data.
    stats = {'mu': data.mean(axis=0), 'sigma': data.std(axis=0)}
    return data_shuffler(data, labels, batch_size=batch_size), stats


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
    """Build a network for instrument classification.

    Returns
    -------
    objective_fx: compiled theano function
        Callable function that takes (x, y, eta) as arguments, returning the
        scalar loss over the data x; implicitly updates the parameters of the
        network given the learning rate eta.
    prediction_fx: compiled theano function
        Callable function that takes (x) as an argument; returns the posterior
        representation for the input data.
    params: dict
        All trainable parameters in the network.
    """
    # ----------------------------------------------------
    # Step 1. Build the network
    # ----------------------------------------------------
    x_input = T.matrix('input')

    # Define layer shapes -- (n_in, n_out)
    l0_dim = (120, 256)
    l1_dim = (256, 256)
    l2_dim = (256, 10)

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

    # Layer 2 - Classifier Layer
    weights2 = theano.shared(np.random.normal(scale=0.01, size=l2_dim),
                             name='weights2')
    bias2 = theano.shared(np.zeros(l2_dim[1]), name='bias2')
    z_output = T.nnet.softmax(T.dot(z_out1, weights2) + bias2)

    # ----------------------------------------------------
    # Step 2. Define a loss function
    # ----------------------------------------------------
    y_target = T.ivector('y_target')
    observation_index = T.arange(y_target.shape[0], dtype='int32')
    scalar_loss = T.mean(-T.log(z_output)[observation_index, y_target])
    accuracy = T.mean(T.eq(T.argmax(z_output, axis=1), y_target))

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
    objective_fx = theano.function(inputs=[x_input, y_target, eta],
                                   outputs=100*(1.0 - accuracy),
                                   updates=updates,
                                   allow_input_downcast=True)

    pred_fx = theano.function(inputs=[x_input],
                              outputs=z_output,
                              allow_input_downcast=True)

    # Add mu and sigma variables now, as we don't want to update them
    # during training.
    network_params.update({mu_obs.name: mu_obs,
                           sigma_obs.name: sigma_obs})
    return objective_fx, pred_fx, network_params


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


def main(args):
    """Main routine for training a deep network.

    After training a deep network some number of iterations, the error over the
    last batch update is reported and the total error over the holdout set is
    computed.

    As a point of reference, sklearn's SVC with a linear kernel achieves train
    and test error of approximately 13%/20%, respectively. Here, with 50k
    iterations and a learning rate of 0.025, the deep net achieves train and
    test error of 2%/6.2%, respectively.

    Parameters
    ----------
    args : ArgumentParser
        Initialized argument object.
    """
    obj_fx, pred_fx, params = build_network()
    shuffler, stats = prepare_training_data(
        args.train_data_file, args.train_label_file, args.batch_size)

    # Set network's mu and sigma values.
    for name in ['mu', 'sigma']:
        params[name].set_value(stats[name])

    errors = train_network(obj_fx,
                           shuffler,
                           args.learning_rate,
                           args.max_iterations,
                           args.print_frequency)

    print "Final Training Error: %0.4f" % errors[-1]
    # Prepare testing data to step through in batches.
    test_data = data_stepper(
        np.load(args.test_data_file), np.load(args.test_label_file), 500)
    test_errors = [np.equal(pred_fx(x_m).argmax(axis=1),
                            y_m) for x_m, y_m in test_data]
    print "Test Error: %0.4f" % (100*(1 - np.concatenate(test_errors).mean()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Learn chroma features from DFT magnitude spectra.")
    parser.add_argument("train_data_file",
                        metavar="train_data_file", type=str,
                        help="Filepath to train data.")
    parser.add_argument("train_label_file",
                        metavar="train_label_file", type=str,
                        help="Filepath to train labels.")
    parser.add_argument("test_data_file",
                        metavar="test_data_file", type=str,
                        help="Filepath to test data.")
    parser.add_argument("test_label_file",
                        metavar="test_label_file", type=str,
                        help="Filepath to test labels.")
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

