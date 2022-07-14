import numpy as np
from absl import app, flags
import tensorflow as tf
from model import retrieval_model
import os
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
import math
import tensorflow_addons as tfa
import numpy as np
from model import Model
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' #This is a tmp fix of cudnn error





def evaluate(model, x, y, max_ratation=0):
    # Let's rotate the images before evaluation by max_ratation
    #
    #
    results = []

    for eps_i in [1, 2, 3, 4, 5, 6, 7, 8]:
        eps = eps_i / 255.0
        eps_iter = eps / 20
        rand_init = eps
        iter_num = int(2 * eps / eps_iter)
        batch = 32
        adv_xs = []
        for i in range((x.shape[0]-1)//batch + 1):
            xx = x[i*batch:min((i+1)*batch, x.shape[0]), :, :, :]
            yy = y[i*batch:min((i+1)*batch, x.shape[0])]
            degrees = (np.random.random(xx.shape[0]) - 0.5) * 2 * max_ratation
            # rotated_xx = tf.contrib.image.rotate(xx, degrees * math.pi / 180, interpolation='BILINEAR')
            rotated_xx = tfa.image.rotate(xx, degrees * math.pi / 180)
            x_pgd = projected_gradient_descent(model, rotated_xx, eps, eps_iter, iter_num, np.inf, y=yy,
                                               rand_init=rand_init)
            adv_xs.append(x_pgd)
        adv_x = np.vstack(adv_xs)
        scores = model.evaluate(adv_x, y, verbose=1)
        seg = '{},{},{},{},{},{}'.format('pgd', eps, eps_iter, iter_num, rand_init, scores[1])
        print(seg)
        results.append(seg)

    return results


def main(_):
    model = retrieval_model('models/cnn_model', 50, False)
    with open('eval_black.log', 'w') as file:
        for ap_eps in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            adv_xs = np.load('data/xzz_500_{}_s.npy'.format(ap_eps))
            ys = np.load('data/yzz_500_{}_s.npy'.format(ap_eps))

            print(adv_xs.shape)
            print(ys.shape)
            ys = ys.reshape((ys.shape[0],))
            results = evaluate(model, adv_xs, ys, 15)
            for seg in results:
                line = '{},{}'.format(ap_eps, seg) + '\n'
                print(line)
                file.write(line)
                file.flush()

if __name__ == "__main__":
    app.run(main)
