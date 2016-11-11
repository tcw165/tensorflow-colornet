import tensorflow as tf


class ClassNet:
    _input = None
    _output = None

    def __init__(self,
                 input_tensor,
                 model_path=None):
        with tf.variable_scope("ClassNet"):
            self._input = input_tensor

            # Init the model.
            self._init_model(model_path)

            # Determine the input size.
            self._input = tf.placeholder(tf.float32, [None, 256])

            # Build the graph.
            # Layer 1.
            hidden_layer = tf.matmul(self._input,
                                     self._model["nn1_w"],
                                     name="nn1_w")
            hidden_layer = tf.nn.bias_add(hidden_layer,
                                          self._model["nn1_w"],
                                          name="nn1_w")
            hidden_layer = tf.nn.relu(hidden_layer,
                                      name="relu1")

            # Layer 2.
            hidden_layer = tf.matmul(hidden_layer,
                                     self._model["nn2_w"],
                                     name="nn2_w")
            hidden_layer = tf.nn.bias_add(hidden_layer,
                                          self._model["nn2_b"],
                                          name="nn2_b")
            self._output = tf.nn.sigmoid(hidden_layer,
                                         name="output")

    @property
    def output(self):
        """
        The output tensor of the network.
        """
        return self._output

    def _init_model(self, model_path=None):
        # Determine the model (either from scratch or load pre-trained data).
        if model_path is None:
            self._model = {
                # H x W x In x Out
                "nn1_w": tf.Variable(tf.truncated_normal([512, 256])),
                "nn1_b": tf.Variable(tf.truncated_normal([256])),
                "nn2_w": tf.Variable(tf.truncated_normal([256, 205])),
                "nn2_b": tf.Variable(tf.truncated_normal([205]))
            }
        else:
            # TODO: Support loading the pre-trained model.
            pass
