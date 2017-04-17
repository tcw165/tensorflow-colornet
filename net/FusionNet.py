import tensorflow as tf


class FusionNet:
    _output = None

    def __init__(self,
                 dynamic_size_input,
                 fixed_size_input):
        """
        A network that fuses the dynamic_size_input and fixed_size_input
        together.

        :param dynamic_size_input: Size of [batch_size, H/8, W/8, 256].
        :param fixed_size_input: Size of [256,]
        """
        with tf.variable_scope("FusionNet"):
            # Init the model.
            self._init_model()

            # input_tensor_1 size is [batch_size, H/8, W/8, 256].
            # input_tensor_2 size is [256].
            # Transform the [256,] tensor to [1,1,1,256] tensor.
            hidden_layer = tf.expand_dims(fixed_size_input, 0)
            hidden_layer = tf.expand_dims(hidden_layer, 0)
            hidden_layer = tf.expand_dims(hidden_layer, 0)
            hidden_layer = tf.expand_dims(hidden_layer, 0)
            # Transform the [1,1,1,256] tensor to [batch_size,H/8,W/8,256]
            # tensor so that it could be concatenated to the dynamic_size_input.
            hidden_layer = tf.tile(hidden_layer,
                                   [dynamic_size_input.get_shape()[0].value,
                                    dynamic_size_input.get_shape()[1].value,
                                    dynamic_size_input.get_shape()[2].value,
                                    1])
            # Fuse the dynamic_size_input and fixed_size_input together, so the
            # channels size becomes 512.
            # Now the tensor is size of [batch_size, H/8, W/8, 512].
            hidden_layer = tf.concat(3, [dynamic_size_input, hidden_layer])
            # Expand the dimension so that the tensor becomes size of
            # [batch_size, H/8, W/8, 512, 1].
            hidden_layer = tf.expand_dims(hidden_layer, -1)
            # Transform the weights so that the tensor becomes
            # size of [batch_size, H/8, W/8, 256, 512].
            weights = self._model["fusion_w"]
            weights = tf.expand_dims(weights, 0)
            weights = tf.expand_dims(weights, 0)
            weights = tf.expand_dims(weights, 0)
            weights = tf.tile(weights,
                              [dynamic_size_input.get_shape()[0].value,
                               dynamic_size_input.get_shape()[1].value,
                               dynamic_size_input.get_shape()[2].value,
                               1,
                               1])
            # Squash the channels of size 512 to size of 256. The size becomes
            # [batch_size, H/8, W/8, 256]
            hidden_layer = tf.batch_matmul(weights,
                                           hidden_layer,
                                           name="mul fusion_w")
            hidden_layer = tf.reshape(hidden_layer,
                                      [dynamic_size_input.get_shape()[0].value,
                                       dynamic_size_input.get_shape()[1].value,
                                       dynamic_size_input.get_shape()[2].value,
                                       256])
            # hidden_layer = tf.batch_matmul()
            hidden_layer = tf.nn.bias_add(hidden_layer,
                                          self._model["fusion_b"],
                                          name="add fusion_b")
            self._output = tf.nn.relu(hidden_layer,
                                      name="relu1")

    @property
    def output(self):
        return self._output

    def _init_model(self, model_path=None):
        # Determine the model (either from scratch or load pre-trained data).
        if model_path is None:
            self._model = {
                # H x W x In x Out
                "fusion_w": tf.Variable(tf.truncated_normal([512, 256])),
                "fusion_b": tf.Variable(tf.truncated_normal([256]))
            }
        else:
            # TODO: Support loading the pre-trained model.
            pass
