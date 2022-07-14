"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
import tensorflow as tf
import numpy as np
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.utils import clip_eta, random_lp_vector
from cleverhans.tf2.utils import optimize_linear, compute_gradient
import tensorflow_addons as tfa
import math

def rpgd_defense(
    model_fn,
    x,
    eps,
    eps_iter,
    nb_iter,
    norm,
    loss_fn=None,
    clip_min=None,
    clip_max=None,
    y=None,
    rand_init=None,
    rand_minmax=None,
):
    if loss_fn is None:
        loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits

    def rev_loss(labels, logits, name='reverse_loss'):
        return tf.math.negative(
            loss_fn(labels, logits), name=name
        )

    # Initialize loop variables
    if rand_minmax is None:
        rand_minmax = eps

    if rand_init:
        eta = random_lp_vector(
            tf.shape(x), norm, tf.cast(rand_minmax, x.dtype), dtype=x.dtype
        )
    else:
        eta = tf.zeros_like(x)

    # Clip eta
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta
    if clip_min is not None or clip_max is not None:
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        y = tf.argmax(model_fn(x), 1)

    i = 0
    while i < nb_iter:
        adv = fast_gradient_method(
            model_fn,
            adv_x,
            eps_iter,
            norm,
            rev_loss,
            clip_min=clip_min,
            clip_max=clip_max,
            y=y,
            targeted=None,
        )

        # Clipping perturbation eta to norm norm ball
        eta = adv - adv_x
        eta = clip_eta(eta, norm, eps)
        adv_x = adv_x + eta
        if clip_min is not None or clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
        i += 1
    return adv_x

def zigzag_defense(
    model_fn,
    x,
    eps,
    eps_iter,
    nb_iter,
    alpha,
    norm,
    loss_fn=None,
    clip_min=None,
    clip_max=None,
    y=None,
    rand_init=None,
    rand_minmax=None,
):
    if loss_fn is None:
        loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        y = tf.argmax(model_fn(x), 1)
    # Initialize loop variables
    if rand_minmax is None:
        rand_minmax = eps

    if rand_init:
        eta = random_lp_vector(
            tf.shape(x), norm, tf.cast(rand_minmax, x.dtype), dtype=x.dtype
        )
    else:
        eta = tf.zeros_like(x)

    # Clip eta
    eta = clip_eta(eta, norm, eps)
    x_start = np.copy(x + eta)
    final_x = np.zeros(shape=x.shape)
    # corrected = set()

    for ii in range(nb_iter):
        x_pgd = projected_gradient_descent(model_fn, x_start, eps, eps_iter, int(2*eps/eps_iter), np.inf, y=y, rand_init=rand_init)
        # y_pred_pgd = model_fn(x_pgd)
        # for iid in range(x.shape[0]):
        #     if iid not in corrected and np.argmax(y_pred_pgd[iid]) == y[iid]:
        #         final_x[iid, :] = x_start[iid, :]
        #         # corrected.add(iid)
        # print('{},{}'.format(ii, len(corrected)))
        grad = compute_gradient(model_fn, loss_fn, x_pgd, y, targeted=False)
        optimal_perturbation = optimize_linear(grad, eps_iter, norm)
        x_start = x_start - grad * alpha
        # x_start = x_start - optimal_perturbation
        # If clipping is needed, reset all values outside of [clip_min, clip_max]
        if (clip_min is not None) or (clip_max is not None):
            # We don't currently support one-sided clipping
            assert clip_min is not None and clip_max is not None
            x_start = tf.clip_by_value(x_start, clip_min, clip_max)
        x_start = np.clip(x_start, 0, 1)
        # x_start = np.clip(x_start, x - eps, x + eps)
        eta = x_start - x
        eta = clip_eta(eta, norm, eps)
        x_start = x + eta
    for iid in range(x.shape[0]):
        # if iid not in corrected:
        final_x[iid, :] = x_start[iid, :]
            # corrected.add(iid)
    return final_x

def zigzag_defense_with_rotation(
    model_fn,
    x,
    eps,
    eps_iter,
    nb_iter,
    alpha,
    norm,
    loss_fn=None,
    clip_min=None,
    clip_max=None,
    y=None,
    rand_init=None,
    rand_minmax=None,
    max_rotation=0,
):
    if loss_fn is None:
        loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        y = tf.argmax(model_fn(x), 1)
    # Initialize loop variables
    if rand_minmax is None:
        rand_minmax = eps

    if rand_init:
        eta = random_lp_vector(
            tf.shape(x), norm, tf.cast(rand_minmax, x.dtype), dtype=x.dtype
        )
    else:
        eta = tf.zeros_like(x)

    # Clip eta
    eta = clip_eta(eta, norm, eps)
    x_start = np.copy(x + eta)
    final_x = np.zeros(shape=x.shape)
    # corrected = set()

    for ii in range(nb_iter):
        degrees = (np.random.random(x_start.shape[0]) - 0.5) * 2 * max_rotation
        rotated_xx = tfa.image.rotate(x_start, degrees * math.pi / 180)
        x_pgd = projected_gradient_descent(model_fn, rotated_xx, eps, eps_iter, int(2*eps/eps_iter), np.inf, y=y, rand_init=rand_init)
        # y_pred_pgd = model_fn(x_pgd)
        # for iid in range(x.shape[0]):
        #     if iid not in corrected and np.argmax(y_pred_pgd[iid]) == y[iid]:
        #         final_x[iid, :] = x_start[iid, :]
        #         # corrected.add(iid)
        # print('{},{}'.format(ii, len(corrected)))
        grad = compute_gradient(model_fn, loss_fn, x_pgd, y, targeted=False)
        optimal_perturbation = optimize_linear(grad, eps_iter, norm)
        x_start = x_start - grad * alpha
        # x_start = x_start - optimal_perturbation
        # If clipping is needed, reset all values outside of [clip_min, clip_max]
        if (clip_min is not None) or (clip_max is not None):
            # We don't currently support one-sided clipping
            assert clip_min is not None and clip_max is not None
            x_start = tf.clip_by_value(x_start, clip_min, clip_max)
        x_start = np.clip(x_start, 0, 1)
        # x_start = np.clip(x_start, x - eps, x + eps)
        eta = x_start - x
        eta = clip_eta(eta, norm, eps)
        x_start = x + eta
    for iid in range(x.shape[0]):
        # if iid not in corrected:
        final_x[iid, :] = x_start[iid, :]
            # corrected.add(iid)
    return final_x