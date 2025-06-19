import tensorflow as tf
from utils.common_functions import silog_loss

@tf.function
def train_step(images, true_depth,STUDENT_MODEL, optimizer, train_metric):
    print(f"train step imges {images.shape} true_depth : {true_depth.shape}")
    with tf.GradientTape() as tape:
        predicted_depth = STUDENT_MODEL(images, training=True)
        epsilon = 1e-6

        # Mask is based on teacher depth
        mask = tf.logical_and(true_depth > epsilon, tf.logical_not(tf.math.is_nan(true_depth)))

        #Clean float tensors
        true_depth_clean = tf.where(tf.math.is_nan(true_depth), epsilon, true_depth)
        predicted_depth = tf.where(tf.math.is_nan(predicted_depth), epsilon, predicted_depth)

        #Compute loss
        total_loss = silog_loss(
            y_true=true_depth_clean,
            y_pred=predicted_depth,
            mask=mask
        )
    grads = tape.gradient(total_loss, STUDENT_MODEL.trainable_variables)
    optimizer.apply_gradients(zip(grads, STUDENT_MODEL.trainable_variables))
    train_metric.update_state(total_loss)
    return total_loss


@tf.function
def val_step(images, true_depth, STUDENT_MODEL, val_metric):
    predicted_depth = STUDENT_MODEL(images, training=False)

    epsilon = 1e-6

    #Mask based on teacher depth
    mask = tf.logical_and(true_depth > epsilon, tf.logical_not(tf.math.is_nan(true_depth)))

    #Clean float tensors
    true_depth_clean = tf.where(tf.math.is_nan(true_depth), epsilon, true_depth)
    predicted_depth = tf.where(tf.math.is_nan(predicted_depth), epsilon, predicted_depth)

    #Compute loss
    total_loss = silog_loss(
        y_true=true_depth_clean,
        y_pred=predicted_depth,
        mask=mask
    )
    val_metric.update_state(total_loss)

    return total_loss