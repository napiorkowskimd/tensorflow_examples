import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


SEQUENCE_LENGTH=4
NUM_OF_FEATURES=16
NUM_OF_EPOCHS = 100
UNITS = 32
use_epochs = False


class RNNBinaryAdder(object):
    """docstring for RNNBinaryAdder."""
    def __init__(self, units, input_sequence_length, input_features):
        super(RNNBinaryAdder, self).__init__()
        self.units = units
        self.input_sequence_length = input_sequence_length
        self.input_features = input_features

        self.cell = tf.nn.rnn_cell.BasicRNNCell(self.units,
                                                activation=tf.identity)
        self.output_layer = tf.layers.Dense(self.input_features,
                                            activation=tf.sigmoid)

    def predict(self, input_sequence):
        """Calculates prediction for next element in sequence
        Arguments:
            input_sequence: tensor with shape [batch_size, sequence_length, num_of_features]
        Returns:
            tensor with shape [batch_size, num_of_features]
        """
        # We need to feed static_rnn with list of tensors, so we unstack
        # input tensor
        elements = tf.unstack(input_sequence, axis=1)
        predictions, _ = tf.nn.static_rnn(self.cell, elements, dtype=tf.float32)
        # we're not interested in intermediate predictions
        predictions = predictions[-1]
        outputs = self.output_layer(predictions)
        return outputs

    def train(self, x, y, learning_rate=0.09):
        predictions = self.predict(x)
        loss = tf.losses.log_loss(y, predictions,
                                reduction=tf.losses.Reduction.NONE)
        vec_length = loss.get_shape().as_list()[-1]
        weights = np.array([2**i for i in range(vec_length, 0, -1)],
                        dtype=np.dtype('float32'))
        weights /= 2**vec_length
        loss = tf.matmul(loss, weights.reshape((-1, 1)))
        loss = tf.reduce_sum(loss)
        fit = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return (loss, fit)


def _int_to_binary_vector(n, length=8):
    assert(n < (1 << length))
    _str = ''.join(['{0:0', str(length), 'b}']).format(n)
    vec = [int(s) for s in _str]
    return np.array(vec)

def _bin_vec_to_int(vec):
    _str = ''.join(str(int(k)) for k in vec.reshape([-1]))
    return int(_str, 2)

def generate_data(sequence_length, vec_length):
    n = (1 << (vec_length-sequence_length+1)) - sequence_length
    x, y = np.zeros((n, sequence_length, vec_length)), np.zeros((n, vec_length))
    for i in range(n):
        x[i, :] = np.array([_int_to_binary_vector(k, vec_length)
                                for k in range(i, i+sequence_length)])
        y[i] = _int_to_binary_vector(sum(range(i, i+sequence_length)),
                                    vec_length)

    return (x.reshape((-1, sequence_length, vec_length)),
            y.reshape((-1, vec_length)))


if __name__ == "__main__":
    X = tf.placeholder(shape=[None, SEQUENCE_LENGTH, NUM_OF_FEATURES],
                    dtype=tf.float32, name="X")
    Y = tf.placeholder(shape=[None, NUM_OF_FEATURES],
                    dtype=tf.float32, name="Y")

    net = RNNBinaryAdder(UNITS, SEQUENCE_LENGTH, NUM_OF_FEATURES)

    train_step = net.train(X, Y, 0.07)
    predictions_step = net.predict(X)


    dataset = generate_data(SEQUENCE_LENGTH, NUM_OF_FEATURES)
    train_idxs = np.random.choice(len(dataset[0]), size=int(0.15*len(dataset[0])))
    NUM_OF_SAMPLES = len(train_idxs)
    train_data = (dataset[0][train_idxs], dataset[1][train_idxs])
    BATCH_SIZE= min((400, NUM_OF_SAMPLES))

    print("number of samples: ", NUM_OF_SAMPLES)
    print("batch size: ", BATCH_SIZE)

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        print("Training.....")
        i = 0
        cumulative_loss = 100
        try:
            while cumulative_loss > 0.01 and (not use_epochs or i < NUM_OF_EPOCHS):
                cumulative_loss = 0
                for j in range(int(NUM_OF_SAMPLES/BATCH_SIZE)):
                    feed_dict = {X: train_data[0][j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                                 Y: train_data[1][j*BATCH_SIZE:(j+1)*BATCH_SIZE]}
                    l, _ = s.run(train_step, feed_dict)
                    cumulative_loss = cumulative_loss + l
                feed_dict = {X: train_data[0][-NUM_OF_SAMPLES%BATCH_SIZE:],
                             Y: train_data[1][-NUM_OF_SAMPLES%BATCH_SIZE:]}
                l, _ = s.run(train_step, feed_dict)
                cumulative_loss = cumulative_loss + l

                print("epoch {}/{}: loss: {:.2f}".format(i+1, NUM_OF_EPOCHS if use_epochs else "inf", cumulative_loss))
                i += 1

        except KeyboardInterrupt:
            pass

        feed_dict = {X: dataset[0]}
        predictions = s.run(predictions_step, feed_dict)

        predicted_numbers = np.array([_bin_vec_to_int(k) for k in np.round(predictions)])
        real_numbers = np.array([_bin_vec_to_int(d) for d in dataset[1]])
        print("accuracy:", np.sum(predicted_numbers == real_numbers)/len(real_numbers))

        plt.plot(real_numbers, 'red')
        plt.plot(predicted_numbers, 'green')
        plt.show()
