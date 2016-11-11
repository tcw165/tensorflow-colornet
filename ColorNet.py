import tensorflow as tf
from net.LowLvFeatNet import LowLvFeatNet
from net.MidLvFeatNet import MidLvFeatNet
from net.GlobalFeatNet import GlobalFeatNet
from net.ClassNet import ClassNet
from net.FusionNet import FusionNet
from net.UpSizeColorNet import UpSizeColorNet


class ColorNet:

    _input_dynamic = None
    _input_fixed = None
    _output = None

    def __init__(self,
                 image_names,
                 input_size,
                 batch_size,
                 mode_path=None):
        with tf.variable_scope("ColorNet"):
            # Determine the input size.
            input_width = input_size[0]
            input_height = input_size[1]
            self._input_dynamic = tf.placeholder(tf.float32,
                                                 shape=[None,
                                                        input_width,
                                                        input_height,
                                                        1])
            self._input_fixed = tf.placeholder(tf.float32,
                                               shape=[None, 224, 224, 1])

            # Init the Low-Level-Feature-Network.
            low_lv_feat_net = LowLvFeatNet(self._input_dynamic,
                                           self._input_fixed)

            # Init the Mid-Level-Feature-Network.
            mid_lv_feat_net = MidLvFeatNet(low_lv_feat_net.dynamic_size_output)

            # Init the Global-Feature-Network.
            global_feat_net = GlobalFeatNet(low_lv_feat_net.fixed_size_output)

            # TODO: Init the Fusion layer.
            fusion_layer = FusionNet(mid_lv_feat_net.output)

            # Init the Class-Network.
            class_net = ClassNet(global_feat_net.output_2)
            # TODO: The loss tensor.

            # Init the Colorization-Network.
            up_size_color_net = UpSizeColorNet(fusion_layer.output)
            # TODO: The output image?

    @property
    def input(self):
        return self._input_dynamic

    @property
    def output(self):
        return self._output
