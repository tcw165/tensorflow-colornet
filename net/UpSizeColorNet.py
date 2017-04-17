import tensorflow as tf
from tensorflow.python.ops.image_ops import ResizeMethod


class UpSizeColorNet:
    _output = None

    def __init__(self,
                 input_tensor):
        with tf.variable_scope("UpSizeColorNet"):
            # Init the model.
            self._init_model()

            # Conv-layer, channels from 256 to 128.
            hidden_layer = tf.nn.conv2d(input_tensor,
                                        self._model["conv1_w"],
                                        strides=[1, 1, 1, 1],
                                        padding="SAME",
                                        data_format="NHWC",
                                        name="conv1_w")
            hidden_layer = tf.nn.bias_add(hidden_layer,
                                          self._model["conv1_b"],
                                          name="conv1_b")
            hidden_layer = tf.nn.relu(hidden_layer,
                                      name="relu1")
            # Up-sample layer, double the size.
            hidden_layer = tf.image.resize_images(
                hidden_layer,
                [2 * hidden_layer.get_shape()[1].value,
                 2 * hidden_layer.get_shape()[2].value],
                method=ResizeMethod.NEAREST_NEIGHBOR)
            # Conv-layer, channels from 128 to 64.
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
            # Conv-layer, channels from 64 to 64.
            hidden_layer = tf.nn.conv2d(hidden_layer,
                                        self._model["conv3_w"],
                                        strides=[1, 1, 1, 1],
                                        padding="SAME",
                                        data_format="NHWC",
                                        name="conv3_w")
            hidden_layer = tf.nn.bias_add(hidden_layer,
                                          self._model["conv3_b"],
                                          name="conv3_b")
            hidden_layer = tf.nn.relu(hidden_layer,
                                      name="relu3")
            # Up-sample layer, double the size.
            hidden_layer = tf.image.resize_images(
                hidden_layer,
                [2 * hidden_layer.get_shape()[1].value,
                 2 * hidden_layer.get_shape()[2].value],
                method=ResizeMethod.NEAREST_NEIGHBOR)
            # Conv-layer, channels from 64 to 32.
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
            # Conv-layer, channels from 32 to 2.
            hidden_layer = tf.nn.conv2d(hidden_layer,
                                        self._model["conv5_w"],
                                        strides=[1, 1, 1, 1],
                                        padding="SAME",
                                        data_format="NHWC",
                                        name="conv5_w")
            hidden_layer = tf.nn.bias_add(hidden_layer,
                                          self._model["conv5_b"],
                                          name="conv5_b")
            self._output = tf.nn.relu(hidden_layer,
                                      name="relu5")

    @property
    def output(self):
        return self._output

    def _init_model(self, model_path=None):
        # Determine the model (either from scratch or load pre-trained data).
        if model_path is None:
            self._model = {
                # H x W x In x Out
                "conv1_w": tf.Variable(tf.truncated_normal([3, 3, 512, 256])),
                "conv1_b": tf.Variable(tf.truncated_normal([256])),
                "conv2_w": tf.Variable(tf.truncated_normal([3, 3, 256, 128])),
                "conv2_b": tf.Variable(tf.truncated_normal([128])),
                "upsample3_w": tf.Variable(tf.truncated_normal([3, 3, 128, 128])),
                "upsample3_b": tf.Variable(tf.truncated_normal([128])),
                "conv4_w": tf.Variable(tf.truncated_normal([3, 3, 128, 64])),
                "conv4_b": tf.Variable(tf.truncated_normal([64])),
                "conv5_w": tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
                "conv5_b": tf.Variable(tf.truncated_normal([64])),
                "upsample6_w": tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
                "upsample6_b": tf.Variable(tf.truncated_normal([64])),
                "conv7_w": tf.Variable(tf.truncated_normal([3, 3, 64, 32])),
                "conv7_b": tf.Variable(tf.truncated_normal([32])),
                "conv8_w": tf.Variable(tf.truncated_normal([3, 3, 32, 2])),
                "conv8_b": tf.Variable(tf.truncated_normal([2]))
            }
        else:
            # TODO: Support loading the pre-trained model.
            pass
