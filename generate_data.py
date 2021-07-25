import numpy as np
from absl import app, flags
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from utils import ld_cifar10
import tensorflow as tf
from model import retrieval_model
import os
from defenses import rpgd_defense, zigzag_defense

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' #This is a tmp fix of cudnn error



def main(_):
    data = ld_cifar10()
    model = retrieval_model('models/res_net', 200, False)

    for eps_i in [1, 2, 3, 4, 5, 6]:
        eps = eps_i / 255.0
        adv_xs = []
        ys = []
        # for x, y in data.test:
        batch = 128
        for i in range(5):
            x, y = data.test
            y = y.reshape((y.shape[0],))
            ind = i * batch
            x = x[ind:(ind + batch),:,:,:]
            y = y[ind:(ind + batch)]
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            y = tf.convert_to_tensor(y, dtype=tf.int32)
            print('sxtian: {}'.format(y.shape))
            x_zigzag = zigzag_defense(model, x, eps, eps/20, 500, eps/20, np.inf, y=y, rand_init=eps/20)
            # x_pgd = x
            # print(x_pgd.shape)
            adv_xs.append(x_zigzag)
            ys.append(y)
        # if i>10:
        #     break
        # i = i+1
        adv_x = np.vstack(adv_xs)
        adv_y = np.vstack(ys)
        np.save('data/xzz_500_{}_s'.format(eps_i), adv_x)
        # np.save('data/yzz_500_{}'.format(eps), adv_y)
        np.save('data/yzz_500_{}_s'.format(eps_i), adv_y.reshape((adv_y.shape[0] * adv_y.shape[1],)))


if __name__ == "__main__":
    flags.DEFINE_integer("nb_epochs", 8, "Number of epochs.")
    flags.DEFINE_float("eps", 0.3, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool(
        "adv_train", False, "Use adversarial training (on PGD adversarial examples)."
    )
    app.run(main)
