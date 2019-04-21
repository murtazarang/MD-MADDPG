import tensorflow as tf

class SharedGRUCell(tf.nn.rnn_cell.GRUCell):
    def __init__(self, num_units, input_size=None, activation=tf.nn.tanh):
        tf.nn.rnn_cell.GRUCell.__init__(self, num_units, input_size, activation)
        self.my_scope = None

    def __call__(self, a, b):
        if self.my_scope == None:
            self.my_scope = tf.get_variable_scope()
        else:
            self.my_scope.reuse_variables()
        return tf.nn.rnn_cell.GRUCell.__call__(self, a, b, self.my_scope)

s