import re
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class Model():
    def __init__(self, time_step, input_size, output_size, rnn_unit, lr):
        self.time_step = time_step
        self.input_size = input_size
        self.output_size = output_size
        self.rnn_unit = rnn_unit
        self.lr = lr

        self.X = tf.placeholder(tf.float32, [None, time_step, input_size])
        self.Y = tf.placeholder(tf.float32, [None, 1])

        self.weights = {
            'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
            'out': tf.Variable(tf.random_normal([rnn_unit, input_size]))
        }
        self.biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
            'out': tf.Variable(tf.constant(0.1, shape=[output_size, ]))
        }

        return

    def network(self, batch_size, layer_num):
        w_in = self.weights['in']
        b_in = self.biases['in']

        input = tf.reshape(self.X, [-1, self.input_size])
        input_rnn = tf.matmul(input, w_in) + b_in
        input_rnn = tf.reshape(input_rnn, [-1, self.time_step, self.rnn_unit])

        cells = [tf.nn.rnn_cell.BasicLSTMCell(self.rnn_unit) for _ in range(layer_num)]

        mcell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        init_state = mcell.zero_state(batch_size=batch_size, dtype=tf.float32)

        output_rnn,final_states = tf.nn.dynamic_rnn(mcell, input_rnn, initial_state=init_state, dtype=tf.float32)

        output = tf.reshape(final_states[-1][1], [-1, self.rnn_unit])
        w_out = self.weights['out']
        b_out = self.biases['out']

        pred = tf.matmul(output, w_out) + b_out

        return pred, final_states