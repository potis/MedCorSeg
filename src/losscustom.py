import logging

import tensorflow as tf

logger = logging.getLogger('rail.metrics.tversky')


def tversky_metric_by_label(y_true, y_pred, alpha=0.3, beta=0.7):
    """
    Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
    -> the score is computed for each class separately and then summed
    alpha=beta=0.5 : dice coefficient
    alpha=beta=1   : tanimoto coefficient (also known as jaccard)
    alpha+beta=1   : produces set of F*-scores
    implemented by E. Moebel, 06/04/18
    """

    ones = tf.ones(tf.shape(input=y_true),  dtype=tf.float32)
    ones=tf.cast(ones, tf.float32)
    y_true=tf.cast(y_true, tf.float32)
    y_pred=tf.cast(y_pred, tf.float32)

    p0 = y_pred  # probability that voxels are class i
    p1 = ones - y_pred  # probability that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    ndim = len(y_pred.shape)
    reduction_axes = list(range(ndim - 1))

    num = tf.keras.backend.sum(p0 * g0, reduction_axes)
    den = num + alpha * tf.keras.backend.sum(p0 * g1, reduction_axes) + beta * tf.keras.backend.sum(p1 * g0,
                                                                                                    reduction_axes)
    # when summing over classes, T has dynamic range [0 Ncl]
    t = num / (den + tf.keras.backend.epsilon())



    return t

def tversky_loss_by_label(y_true, y_pred, alpha=0.3, beta=0.7):
    return 1.0 - tversky_metric_by_label(y_true, y_pred,alpha=alpha, beta=beta)


def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7):
    """
    Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
    -> the score is computed for each class separately and then summed

    alpha=beta=0.5 : dice coefficient
    alpha=beta=1   : tanimoto coefficient (also known as jaccard)
    alpha+beta=1   : produces set of F*-scores
    implemented by E. Moebel, 06/04/18
    """
    loss = tversky_loss_by_label(y_true, y_pred, alpha=alpha, beta=beta)
    loss = tf.reduce_sum ( input_tensor=loss )


    return loss



def dice_nogb(y_true, y_pred):
    # tf.print(tf.shape(y_true))
    # tf.print(tf.shape(y_pred))

    return - tf.keras.backend.mean(dice_coefficient_class(y_true[..., 1:], y_pred[..., 1:]))

def dice_coefficient_class(y_true, y_pred, smooth=1.):
    ndim = len(y_pred.shape)
    reduction_axes = list(range(ndim - 1))
    _epsilon = 10 ** -7
    y_true = tf.keras.backend.cast(y_true, dtype=tf.float32)
    y_pred = tf.keras.backend.cast(y_pred, dtype=tf.float32)
    # Determine axes to pass to tf.sum
    # Calculate intersections and unions per class
    intersections = tf.keras.backend.sum(y_true * y_pred, reduction_axes)
    unions = tf.keras.backend.sum(y_true + y_pred, reduction_axes)
    # Calculate Dice scores per class
    dice_scores = 2.0 * (intersections) / (unions + _epsilon)

    if (tf.reduce_sum(y_true) == 0) and (tf.reduce_sum(y_pred) == 0):
        dice_scores = 1.0


    return dice_scores
