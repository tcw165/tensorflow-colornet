import numpy as np
import tensorflow as tf


class MidLvFeatNet:
    _model = None

    _output = None

    def __init__(self,
                 input_tensor,
                 model_path=None):
        with tf.variable_scope("MidLvFeatNet"):
            # Init the model.
            self._init_model(model_path)

            # Build the graph.
            hidden_layer = tf.nn.conv2d(input_tensor,
                                        self._model["conv1_w"],
                                        strides=[1, 1, 1, 1],
                                        padding="SAME",
                                        data_format="NHWC",
                                        name="conv1_w")
            hidden_layer = tf.nn.bias_add(hidden_layer,
                                          self._model["conv1_b"],
                                          name="conv1_b")
            # TODO: Not sure to use ReLu or Sigmoid transfer function.
            hidden_layer = tf.nn.sigmoid(hidden_layer,
                                         name="sigmoid1")

            hidden_layer = tf.nn.conv2d(hidden_layer,
                                        self._model["conv2_w"],
                                        strides=[1, 1, 1, 1],
                                        padding="SAME",
                                        data_format="NHWC",
                                        name="conv2_w")
            hidden_layer = tf.nn.bias_add(hidden_layer,
                                          self._model["conv2_b"],
                                          name="conv2_b")
            # TODO: Not sure to use ReLu or Sigmoid transfer function.
            self._output = tf.nn.sigmoid(hidden_layer,
                                         name="sigmoid2")

    @property
    def output(self):
        return self._output

    def _init_model(self, model_path=None):
        # Determine the model (either from scratch or load pre-trained data).
        if model_path is None:
            self._model = {
                # H x W x In x Out
                "conv1_w": tf.Variable(tf.truncated_normal([3, 3, 512, 512])),
                "conv1_b": tf.Variable(tf.truncated_normal([512])),
                "conv2_w": tf.Variable(tf.truncated_normal([3, 3, 512, 256])),
                "conv2_b": tf.Variable(tf.truncated_normal([256]))
            }
        else:
            # TODO: Support loading the pre-trained model.
            pass
