import numpy as np
from absl import app, flags
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent

import tensorflow as tf
from model import retrieval_model
import os
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method


from model import Model
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' #This is a tmp fix of cudnn error





def evaluate(model, x, y):
    results = []

    # PGD : k = 10
    for eps in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
        eps_iter = eps / 20
        rand_init = eps
        iter_num = int(2 * eps / eps_iter)
        x_pgd = projected_gradient_descent(model, x, eps, eps_iter, iter_num, np.inf, y=y,
                                           rand_init=rand_init)
        scores = model.evaluate(x_pgd, y, verbose=1)
        seg = '{},{},{},{},{},{}'.format('pgd', eps, eps_iter, iter_num, rand_init, scores[1])
        print(seg)
        results.append(seg)

    return results


def main(_):
    model = retrieval_model('models/cnn_model', 50, False)
    with open('eval_black.log', 'w') as file:
        for ap_eps in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            adv_xs = np.load('data/xzz_500_{}.npy'.format(ap_eps))
            ys = np.load('data/yzz_500_{}.npy'.format(ap_eps))

            print(adv_xs.shape)
            print(ys.shape)
            ys = ys.reshape((ys.shape[0],))
            results = evaluate(model, adv_xs, ys)
            for seg in results:
                line = '{},{}'.format(ap_eps, seg) + '\n'
                print(line)
                file.write(line)
                file.flush()

if __name__ == "__main__":
    app.run(main)
