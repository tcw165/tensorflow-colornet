import numpy as np
import tensorflow as tf


class GlobalFeatNet:
    _model = None

    _output_512 = None
    _output_256 = None

    def __init__(self,
                 input_tensor,
                 model_path=None):
        if __name__ == '__main__':
            with tf.variable_scope("GlobalFeatNet"):
                # Init the model.
                self._init_model(model_path)

                # Build the graph.
                # C-layer 1.
                hidden_layer = tf.nn.conv2d(input_tensor,
                                            self._model["conv1_w"],
                                            strides=[2, 2, 1, 1],
                                            padding="SAME",
                                            data_format="NHWC",
                                            name="conv1_w")
                hidden_layer = tf.nn.bias_add(hidden_layer,
                                              self._model["conv1_b"],
                                              name="conv1_b")
                hidden_layer = tf.nn.relu(hidden_layer,
                                          name="relu1")

                # C-layer 2.
                hidden_layer = tf.nn.conv2d(hidden_layer,
                                            self._model["conv2_w"],
                                            strides=[1, 1, 1, 1],
                                            padding="SAME",
                                            data_format="NHWC",
                                            name="conv2_w")
                hidden_layer = tf.nn.bias_add(hidden_layer,
                                              self._model["conv2_b"],
                                              name="conv2_b")
                hidden_layer = tf.nn.relu(hidden_layer,
                                          name="relu2")

                # C-layer 3.
                hidden_layer = tf.nn.conv2d(hidden_layer,
                                            self._model["conv3_w"],
                                            strides=[2, 2, 1, 1],
                                            padding="SAME",
                                            data_format="NHWC",
                                            name="conv3_w")
                hidden_layer = tf.nn.bias_add(hidden_layer,
                                              self._model["conv3_b"],
                                              name="conv3_b")
                hidden_layer = tf.nn.relu(hidden_layer,
                                          name="relu3")

                # C-layer 4.
                hidden_layer = tf.nn.conv2d(hidden_layer,
                                            self._model["conv4_w"],
                                            strides=[1, 1, 1, 1],
                                            padding="SAME",
                                            data_format="NHWC",
                                            name="conv4_w")
                hidden_layer = tf.nn.bias_add(hidden_layer,
                                              self._model["conv4_b"],
                                              name="conv4_b")
                hidden_layer = tf.nn.relu(hidden_layer,
                                          name="relu4")

                # Ready to the FC layers and squash the n-d data to vector.
                shape = hidden_layer.get_shape().as_list()
                dim = 1
                # The 1st dimension is the batch size.
                for d in shape[1:]:
                    dim *= d
                hidden_layer = tf.reshape(hidden_layer, [-1, dim])

                # FC layer 5.
                hidden_layer = tf.matmul(hidden_layer,
                                         self._model["fc5_w"],
                                         name="fc5_w")
                hidden_layer = tf.nn.bias_add(hidden_layer,
                                              self._model["fc5_b"],
                                              name="fc5_b")
                hidden_layer = tf.nn.relu(hidden_layer,
                                          name="relu5")

                # FC layer 6.
                hidden_layer = tf.matmul(hidden_layer,
                                         self._model["fc6_w"],
                                         name="fc6_w")
                hidden_layer = tf.nn.bias_add(hidden_layer,
                                              self._model["fc6_b"],
                                              name="fc6_b")
                self._output_512 = tf.nn.relu(hidden_layer,
                                              name="relu6")

                # FC layer 7.
                hidden_layer = tf.matmul(self._output_512,
                                         self._model["fc7_w"],
                                         name="fc7_w")
                hidden_layer = tf.nn.bias_add(hidden_layer,
                                              self._model["fc7_b"],
                                              name="fc7_b")
                self._output_256 = tf.nn.relu(hidden_layer,
                                              name="relu7")

    @property
    def output_256(self):
        """
        The last layer of the network.
        """
        return self._output_256

    @property
    def output_512(self):
        """
        The layer before the last layer of the network.
        """
        return self._output_512

    def _init_model(self, model_path=None):
        # Determine the model (either from scratch or load pre-trained data).
        if model_path is None:
            self._model = {
                # H x W x In x Out
                "conv1_w": tf.Variable(tf.truncated_normal([3, 3, 512, 512])),
                "conv1_b": tf.Variable(tf.truncated_normal([512])),
                "conv2_w": tf.Variable(tf.truncated_normal([3, 3, 512, 512])),
                "conv2_b": tf.Variable(tf.truncated_normal([512])),
                "conv3_w": tf.Variable(tf.truncated_normal([3, 3, 512, 512])),
                "conv3_b": tf.Variable(tf.truncated_normal([512])),
                "conv4_w": tf.Variable(tf.truncated_normal([3, 3, 512, 512])),
                "conv4_b": tf.Variable(tf.truncated_normal([512])),
                "fc5_w": tf.Variable(tf.truncated_normal([7 * 7 * 512, 1024])),
                "fc5_b": tf.Variable(tf.truncated_normal([1024])),
                "fc6_w": tf.Variable(tf.truncated_normal([1024, 512])),
                "fc6_b": tf.Variable(tf.truncated_normal([512])),
                "fc7_w": tf.Variable(tf.truncated_normal([512, 256])),
                "fc7_b": tf.Variable(tf.truncated_normal([256]))
            }
        else:
            # TODO: Support loading the pre-trained model.
            pass
