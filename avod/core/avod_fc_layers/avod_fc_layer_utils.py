import tensorflow as tf
from tensorflow.contrib import slim

OFFSETS_OUTPUT_SIZE = {
    'box_3d': 6,
    'box_8c': 24,
    'box_8co': 24,
    'box_4c': 10,
    'box_4ca': 10,
}

ANG_VECS_OUTPUT_SIZE = {
    'box_3d': 2,
    'box_8c': 0,
    'box_8co': 0,
    'box_4c': 0,
    'box_4ca': 2,
}


def feature_fusion(fusion_method, inputs, input_weights,is_training=True):
    """Applies feature fusion to multiple inputs

    Args:
        fusion_method: 'mean' or 'concat'
        inputs: Input tensors of shape (batch_size, width, height, depth)
            If fusion_method is 'mean', inputs must have same dimensions.
            If fusion_method is 'concat', width and height must be the same.
        input_weights: Weight of each input if using 'mean' fusion method

    Returns:
        fused_features: Features after fusion
    """

    # Feature map fusion
    with tf.variable_scope('fusion'):
        fused_features = None

        if fusion_method == 'mean':
            rois_sum = tf.reduce_sum(inputs, axis=0)
            rois_mean = tf.divide(rois_sum, tf.reduce_sum(input_weights))
            fused_features = rois_mean

        elif fusion_method == 'concat':
            # Concatenate along last axis
            last_axis = len(inputs[0].get_shape()) - 1
            fused_features = tf.concat(inputs, axis=last_axis)

        elif fusion_method == 'max':
            fused_features = tf.maximum(inputs[0], inputs[1])

        elif fusion_method == 'lel':
            inputs_shape = inputs[0].get_shape()
            last_axis = len(inputs_shape) - 1
            fused_features = tf.concat(inputs, axis=last_axis)
            out_depth = max(inputs[0].get_shape()[last_axis],
                            inputs[1].get_shape()[last_axis])
            fused_features = slim.conv2d(
                fused_features,
                out_depth, [1, 1],
                scope='1x1_lel',
                normalizer_fn=slim.batch_norm,
                normalizer_params={
                    'is_training': is_training},
                weights_regularizer=tf.contrib.layers.l1_regularizer(scale=0.01))

        else:
            raise ValueError('Invalid fusion method', fusion_method)

    return fused_features
