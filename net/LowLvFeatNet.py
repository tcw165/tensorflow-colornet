import numpy as np
import tensorflow as tf


class LowLvFeatNet:
    _model = None

    _output_dynamic = None
    _output_fixed = None

    def __init__(self,
                 dynamic_size_input,
                 fixed_size_input,
                 model_path=None):
        """
        There're two Low-Level-Feature networks shared the same weights and
        biases in this component. The only difference is that one network takes
        224x224 input image and the other takes WxH input image.

        :param dynamic_size_input: The input tensor with original photo size for
                                   the feed-forward process.
        :param fixed_size_input: The input tensor with fixed size of 224x244 for
                                 the training process.
        :param model_path: The path of the pre-trained model. If the path is not
                           present and the network will use a default weights
                           and biases setting.
        """
        with tf.variable_scope("LowLvFeatNet"):
            # Init the model.
            self._init_model(model_path)

            # Use the shared weights and biases to build the graph.
            self._output_fixed = self._build_graph(dynamic_size_input)
            self._output_dynamic = self._build_graph(fixed_size_input)

    @property
    def fixed_size_output(self):
        """
        :return: The output tensor with fixed size.
        """
        return self._output_fixed

    @property
    def dynamic_size_output(self):
        """
        :return: The output tensor with dynamic size.
        """
        return self._output_dynamic

    def save_to_file(self, path):
        """
        Save the weights and biases to a physics file.
        """
        # TODO: Implement the saving codes.
        pass

    def _init_model(self, model_path=None):
        # Determine the model (either from scratch or load pre-trained data).
        if model_path is None:
            self._model = {
                # H x W x In x Out
                "conv1_w": tf.Variable(tf.truncated_normal([3, 3, 1, 64])),
                "conv1_b": tf.Variable(tf.truncated_normal([64])),
                "conv2_w": tf.Variable(tf.truncated_normal([3, 3, 64, 128])),
                "conv2_b": tf.Variable(tf.truncated_normal([128])),
                "conv3_w": tf.Variable(tf.truncated_normal([3, 3, 128, 128])),
                "conv3_b": tf.Variable(tf.truncated_normal([128])),
                "conv4_w": tf.Variable(tf.truncated_normal([3, 3, 128, 256])),
                "conv4_b": tf.Variable(tf.truncated_normal([256])),
                "conv5_w": tf.Variable(tf.truncated_normal([3, 3, 256, 256])),
                "conv5_b": tf.Variable(tf.truncated_normal([256])),
                "conv6_w": tf.Variable(tf.truncated_normal([3, 3, 256, 512])),
                "conv6_b": tf.Variable(tf.truncated_normal([512]))
            }
        else:
            # TODO: Support loading the pre-trained model.
            pass

    def _build_graph(self, input_tensor):
        # C-layer 1.
        hidden_layer = tf.nn.conv2d(input_tensor,
                                    self._model["conv1_w"],
                                    strides=[1, 2, 2, 1],
                                    padding="SAME",
                                    data_format="NHWC",
                                    name="conv1_w")
        hidden_layer = tf.nn.bias_add(hidden_layer,
                                      self._model["conv1_b"],
                                      name="conv1_b")
        # TODO: Not sure to use ReLu or Sigmoid transfer function.
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
        # TODO: Not sure to use ReLu or Sigmoid transfer function.
        hidden_layer = tf.nn.relu(hidden_layer,
                                  name="relu2")

        # C-layer 3.
        hidden_layer = tf.nn.conv2d(hidden_layer,
                                    self._model["conv3_w"],
                                    strides=[1, 2, 2, 1],
                                    padding="SAME",
                                    data_format="NHWC",
                                    name="conv3_w")
        hidden_layer = tf.nn.bias_add(hidden_layer,
                                      self._model["conv3_b"],
                                      name="conv3_b")
        # TODO: Not sure to use ReLu or Sigmoid transfer function.
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
        # TODO: Not sure to use ReLu or Sigmoid transfer function.
        hidden_layer = tf.nn.relu(hidden_layer,
                                  name="relu4")

        # C-layer 5.
        hidden_layer = tf.nn.conv2d(hidden_layer,
                                    self._model["conv5_w"],
                                    strides=[1, 2, 2, 1],
                                    padding="SAME",
                                    data_format="NHWC",
                                    name="conv5_w")
        hidden_layer = tf.nn.bias_add(hidden_layer,
                                      self._model["conv5_b"],
                                      name="conv5_b")
        # TODO: Not sure to use ReLu or Sigmoid transfer function.
        hidden_layer = tf.nn.relu(hidden_layer,
                                  name="relu5")

        # C-layer 6.
        hidden_layer = tf.nn.conv2d(hidden_layer,
                                    self._model["conv6_w"],
                                    strides=[1, 1, 1, 1],
                                    padding="SAME",
                                    data_format="NHWC",
                                    name="conv6_w")
        hidden_layer = tf.nn.bias_add(hidden_layer,
                                      self._model["conv6_b"],
                                      name="conv6_b")
        # TODO: Not sure to use ReLu or Sigmoid transfer function.
        output = tf.nn.relu(hidden_layer,
                            name="relu6")
        return output
