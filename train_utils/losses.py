import tensorflow as tf


def crossentropy_loss(label, prediction):
    epsilon = tf.constant(0.00001)
    log_pred = tf.math.log(prediction + epsilon, name='Prediction_Log')
    log_pred_2 = tf.math.log(1-prediction + epsilon, name='1-Prediction_Log')
    cross_entropy = -tf.multiply(label, log_pred) - tf.multiply((1-label), log_pred_2)
    return tf.math.reduce_mean(
        cross_entropy
    )
