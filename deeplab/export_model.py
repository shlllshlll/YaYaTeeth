# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Exports trained model to TensorFlow frozen graph."""

import re
import os
import tensorflow as tf

from tensorflow.core.protobuf import saver_pb2
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util
from tensorflow.python.training import checkpoint_management
from tensorflow.python.framework import importer
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import loader
from tensorflow.python import pywrap_tensorflow

from deeplab import common
from deeplab import input_preprocess
from deeplab import model
from deeplab.utils import gpu

slim = tf.contrib.slim
flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_path', None, 'Checkpoint path')

flags.DEFINE_string('export_path', None,
                    'Path to output Tensorflow frozen graph.')

flags.DEFINE_integer('num_classes', 21, 'Number of classes.')

flags.DEFINE_multi_integer('crop_size', [513, 513],
                           'Crop size [height, width].')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 8,
                     'The ratio of input to output spatial resolution.')

# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale inference.
flags.DEFINE_multi_float('inference_scales', [1.0],
                         'The scales to resize images for inference.')

flags.DEFINE_bool('add_flipped_images', False,
                  'Add flipped images during inference or not.')

flags.DEFINE_integer(
    'quantize_delay_step', -1,
    'Steps to start quantized training. If < 0, will not quantize model.')

flags.DEFINE_bool('save_inference_graph', False,
                  'Save inference graph in text proto.')

# Input name of the exported model.
_INPUT_NAME = 'ImageTensor'

# Output name of the exported predictions.
_OUTPUT_NAME = 'SemanticPredictions'
_RAW_OUTPUT_NAME = 'RawSemanticPredictions'

# Output name of the exported probabilities.
_OUTPUT_PROB_NAME = 'SemanticProbabilities'
_RAW_OUTPUT_PROB_NAME = 'RawSemanticProbabilities'


def _create_input_tensors():
    """Creates and prepares input tensors for DeepLab model.

    This method creates a 4-D uint8 image tensor 'ImageTensor' with shape
    [1, None, None, 3]. The actual input tensor name to use during inference is
    'ImageTensor:0'.

    Returns:
      image: Preprocessed 4-D float32 tensor with shape [1, crop_height,
        crop_width, 3].
      original_image_size: Original image shape tensor [height, width].
      resized_image_size: Resized image shape tensor [height, width].
    """
    # input_preprocess takes 4-D image tensor as input.
    input_image = tf.placeholder(
        tf.uint8, [1, None, None, 3], name=_INPUT_NAME)
    original_image_size = tf.shape(input_image)[1:3]

    # Squeeze the dimension in axis=0 since `preprocess_image_and_label` assumes
    # image to be 3-D.
    image = tf.squeeze(input_image, axis=0)
    resized_image, image, _ = input_preprocess.preprocess_image_and_label(
        image,
        label=None,
        crop_height=FLAGS.crop_size[0],
        crop_width=FLAGS.crop_size[1],
        min_resize_value=FLAGS.min_resize_value,
        max_resize_value=FLAGS.max_resize_value,
        resize_factor=FLAGS.resize_factor,
        is_training=False,
        model_variant=FLAGS.model_variant)
    resized_image_size = tf.shape(resized_image)[:2]

    # Expand the dimension in axis=0, since the following operations assume the
    # image to be 4-D.
    image = tf.expand_dims(image, 0)

    return image, original_image_size, resized_image_size


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Prepare to export model to: %s', FLAGS.export_path)

    with tf.Graph().as_default():
        image, image_size, resized_image_size = _create_input_tensors()

        model_options = common.ModelOptions(
            outputs_to_num_classes={common.OUTPUT_TYPE: FLAGS.num_classes},
            crop_size=FLAGS.crop_size,
            atrous_rates=FLAGS.atrous_rates,
            output_stride=FLAGS.output_stride)

        if tuple(FLAGS.inference_scales) == (1.0,):
            tf.logging.info('Exported model performs single-scale inference.')
            predictions = model.predict_labels(
                image,
                model_options=model_options,
                image_pyramid=FLAGS.image_pyramid)
        else:
            tf.logging.info('Exported model performs multi-scale inference.')
            if FLAGS.quantize_delay_step >= 0:
                raise ValueError(
                    'Quantize mode is not supported with multi-scale test.')
            predictions = model.predict_labels_multi_scale(
                image,
                model_options=model_options,
                eval_scales=FLAGS.inference_scales,
                add_flipped_images=FLAGS.add_flipped_images)
        raw_predictions = tf.identity(
            tf.cast(predictions[common.OUTPUT_TYPE], tf.float32),
            _RAW_OUTPUT_NAME)
        raw_probabilities = tf.identity(
            predictions[common.OUTPUT_TYPE + model.PROB_SUFFIX],
            _RAW_OUTPUT_PROB_NAME)

        # Crop the valid regions from the predictions.
        semantic_predictions = raw_predictions[
            :, :resized_image_size[0], :resized_image_size[1]]
        semantic_probabilities = raw_probabilities[
            :, :resized_image_size[0], :resized_image_size[1]]

        # Resize back the prediction to the original image size.
        def _resize_label(label, label_size):
            # Expand dimension of label to [1, height, width, 1] for resize operation.
            label = tf.expand_dims(label, 3)
            resized_label = tf.image.resize_images(
                label,
                label_size,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                align_corners=True)
            return tf.cast(tf.squeeze(resized_label, 3), tf.int32)
        semantic_predictions = _resize_label(semantic_predictions, image_size)
        semantic_predictions = tf.identity(
            semantic_predictions, name=_OUTPUT_NAME)

        semantic_probabilities = tf.image.resize_bilinear(
            semantic_probabilities, image_size, align_corners=True,
            name=_OUTPUT_PROB_NAME)

        if FLAGS.quantize_delay_step >= 0:
            contrib_quantize.create_eval_graph()

        saver = tf.train.Saver(tf.all_variables())

        dirname = os.path.dirname(FLAGS.export_path)
        tf.gfile.MakeDirs(dirname)
        graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
        freeze_graph_with_def_protos(
            graph_def,
            saver.as_saver_def(),
            FLAGS.checkpoint_path,
            _OUTPUT_NAME + ',' + _OUTPUT_PROB_NAME,
            restore_op_name=None,
            filename_tensor_name=None,
            output_graph=FLAGS.export_path,
            clear_devices=True,
            initializer_nodes=None)

        if FLAGS.save_inference_graph:
            tf.train.write_graph(graph_def, dirname, 'inference_graph.pbtxt')


def _has_no_variables(sess):
    """Determines if the graph has any variables.

    Args:
    sess: TensorFlow Session.

    Returns:
    Bool.
    """
    for op in sess.graph.get_operations():
        if op.type.startswith("Variable") or op.type.endswith("VariableOp"):
            return False
    return True


def freeze_graph_with_def_protos(input_graph_def,
                                 input_saver_def,
                                 input_checkpoint,
                                 output_node_names,
                                 restore_op_name,
                                 filename_tensor_name,
                                 output_graph,
                                 clear_devices,
                                 initializer_nodes,
                                 variable_names_whitelist="",
                                 variable_names_blacklist="",
                                 input_meta_graph_def=None,
                                 input_saved_model_dir=None,
                                 saved_model_tags=None,
                                 checkpoint_version=saver_pb2.SaverDef.V2):
    """Converts all variables in a graph and checkpoint into constants.

    Args:
      input_graph_def: A `GraphDef`.
      input_saver_def: A `SaverDef` (optional).
      input_checkpoint: The prefix of a V1 or V2 checkpoint, with V2 taking
        priority.  Typically the result of `Saver.save()` or that of
        `tf.train.latest_checkpoint()`, regardless of sharded/non-sharded or
        V1/V2.
      output_node_names: The name(s) of the output nodes, comma separated.
      restore_op_name: Unused.
      filename_tensor_name: Unused.
      output_graph: String where to write the frozen `GraphDef`.
      clear_devices: A Bool whether to remove device specifications.
      initializer_nodes: Comma separated string of initializer nodes to run before
                         freezing.
      variable_names_whitelist: The set of variable names to convert (optional, by
                                default, all variables are converted).
      variable_names_blacklist: The set of variable names to omit converting
                                to constants (optional).
      input_meta_graph_def: A `MetaGraphDef` (optional),
      input_saved_model_dir: Path to the dir with TensorFlow 'SavedModel' file
                             and variables (optional).
      saved_model_tags: Group of comma separated tag(s) of the MetaGraphDef to
                        load, in string format (optional).
      checkpoint_version: Tensorflow variable file format (saver_pb2.SaverDef.V1
                          or saver_pb2.SaverDef.V2)

    Returns:
      Location of the output_graph_def.
    """
    del restore_op_name, filename_tensor_name  # Unused by updated loading code.

    # 'input_checkpoint' may be a prefix if we're using Saver V2 format
    if (not input_saved_model_dir and
            not checkpoint_management.checkpoint_exists(input_checkpoint)):
        raise ValueError("Input checkpoint '" + input_checkpoint +
                         "' doesn't exist!")

    if not output_node_names:
        raise ValueError(
            "You need to supply the name of a node to --output_node_names.")

    # Remove all the explicit device specifications for this node. This helps to
    # make the graph more portable.
    if clear_devices:
        if input_meta_graph_def:
            for node in input_meta_graph_def.graph_def.node:
                node.device = ""
        elif input_graph_def:
            for node in input_graph_def.node:
                node.device = ""

    if input_graph_def:
        _ = importer.import_graph_def(input_graph_def, name="")

    with tf.Session() as sess:
        if input_saver_def:
            saver = saver_lib.Saver(
                saver_def=input_saver_def, write_version=checkpoint_version)
            saver.restore(sess, input_checkpoint)
        elif input_meta_graph_def:
            restorer = saver_lib.import_meta_graph(
                input_meta_graph_def, clear_devices=True)
            restorer.restore(sess, input_checkpoint)
            if initializer_nodes:
                sess.run(initializer_nodes.replace(" ", "").split(","))
        elif input_saved_model_dir:
            if saved_model_tags is None:
                saved_model_tags = []
            loader.load(sess, saved_model_tags, input_saved_model_dir)
        else:
            var_list = {}
            reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
            var_to_shape_map = reader.get_variable_to_shape_map()

            # List of all partition variables. Because the condition is heuristic
            # based, the list could include false positives.
            all_parition_variable_names = [
                tensor.name.split(":")[0]
                for op in sess.graph.get_operations()
                for tensor in op.values()
                if re.search(r"/part_\d+/", tensor.name)
            ]
            has_partition_var = False

            for key in var_to_shape_map:
                try:
                    tensor = sess.graph.get_tensor_by_name(key + ":0")
                    if any(key in name for name in all_parition_variable_names):
                        has_partition_var = True
                except KeyError:
                    # This tensor doesn't exist in the graph (for example it's
                    # 'global_step' or a similar housekeeping element) so skip it.
                    continue
                var_list[key] = tensor

            try:
                saver = saver_lib.Saver(
                    var_list=var_list, write_version=checkpoint_version)
            except TypeError as e:
                # `var_list` is required to be a map of variable names to Variable
                # tensors. Partition variables are Identity tensors that cannot be
                # handled by Saver.
                if has_partition_var:
                    raise ValueError(
                        "Models containing partition variables cannot be converted "
                        "from checkpoint files. Please pass in a SavedModel using "
                        "the flag --input_saved_model_dir.")
                # Models that have been frozen previously do not contain Variables.
                elif _has_no_variables(sess):
                    raise ValueError(
                        "No variables were found in this model. It is likely the model "
                        "was frozen previously. You cannot freeze a graph twice.")
                    return 0
                else:
                    raise e

            saver.restore(sess, input_checkpoint)
            if initializer_nodes:
                sess.run(initializer_nodes.replace(" ", "").split(","))

        variable_names_whitelist = (
            variable_names_whitelist.replace(" ", "").split(",")
            if variable_names_whitelist else None)
        variable_names_blacklist = (
            variable_names_blacklist.replace(" ", "").split(",")
            if variable_names_blacklist else None)

        if input_meta_graph_def:
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_meta_graph_def.graph_def,
                output_node_names.replace(" ", "").split(","),
                variable_names_whitelist=variable_names_whitelist,
                variable_names_blacklist=variable_names_blacklist)
        else:
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                output_node_names.replace(" ", "").split(","),
                variable_names_whitelist=variable_names_whitelist,
                variable_names_blacklist=variable_names_blacklist)

    # Write GraphDef to file if output path has been given.
    if output_graph:
        with gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

    return output_graph_def


if __name__ == '__main__':
    # gpu.set_gpu_growth(1)
    # gpu.force_gpu_growth()
    flags.mark_flag_as_required('checkpoint_path')
    flags.mark_flag_as_required('export_path')
    tf.app.run()
