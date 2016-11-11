import tensorflow as tf

class FusionNet:

    _output = None

    def __init__(self,
                 input_tensor):
        pass

    @property
    def output(self):
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
