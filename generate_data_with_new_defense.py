import numpy as np
from absl import app, flags
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from utils import ld_mnist

from model import CNNModel, retrieval_model
import tensorflow as tf
import os
from defenses import zigzag_defense_with_rotation

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' #This is a tmp fix of cudnn error



def main(_):
    data = ld_mnist()
    model = retrieval_model('models/cnn_model', 50, False)

    for eps in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
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
            if eps > 0:
                x_zigzag = zigzag_defense_with_rotation(model, x, eps, eps/20, 500, eps/20, np.inf, y=y, rand_init=eps/20)
            else:
                x_zigzag = x

            adv_xs.append(x_zigzag)
            ys.append(y)

        adv_x = np.vstack(adv_xs)
        adv_y = np.vstack(ys)
        np.save('data/xzz_500_{}_s'.format(eps_i), adv_x)
        # np.save('data/yzz_500_{}'.format(eps), adv_y)
        np.save('data/yzz_500_{}_s'.format(eps_i), adv_y.reshape((adv_y.shape[0] * adv_y.shape[1],)))


if __name__ == "__main__":
    app.run(main)
