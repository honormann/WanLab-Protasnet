import tensorflow as tf
import keras.backend as K

def neg_par_log_likelihood_loss(y_true, score):

    """
    Args
        score: 		predicted survival time, tf tensor of shape (None, 1)
        time:		true survival time, tf tensor of shape (None, )
        event:		event, tf tensor of shape (None, )
    Return
        loss:		negative partial likelihood of cox regression
    """

    event, time = y_true[:, 0], y_true[:, 1]

    n_observed = tf.reduce_sum(event)

    ## find index i satisfying event[i]==1
    ix = tf.where(tf.cast(event, tf.bool))  # shape of ix is [None, 1]

    ## sel_mat is a matrix where sel_mat[i,j]==1 where time[i]<=time[j]
    ytime_indicator = tf.cast(tf.expand_dims(time, -1) <= time, tf.float32)

    risk_set_sum = K.dot(ytime_indicator, (tf.exp(score)))
    diff = score - tf.math.log(risk_set_sum)
    sum_diff_in_observed = tf.reduce_sum(tf.gather(diff, ix))
    cost = - (sum_diff_in_observed / n_observed)

    return cost