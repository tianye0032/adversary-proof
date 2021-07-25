import numpy as np
from absl import app, flags
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from utils import ld_mnist

from model import CNNModel, retrieval_model
import os
from defenses import rpgd_defense, zigzag_defense

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' #This is a tmp fix of cudnn error



def main(_):
    data = ld_mnist()
    model = retrieval_model('models/cnn_model', 50, False)

    for eps in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        adv_xs = []
        ys = []
        i = 0
        for x, y in data.test:
            x_zigzag = zigzag_defense(model, x, eps, eps/40, 500, eps/40, np.inf, y=y, rand_init=eps/40)
            adv_xs.append(x_zigzag)
            ys.append(y)
            if i > 10:
                break
            i = i+1
        adv_x = np.vstack(adv_xs)
        adv_y = np.vstack(ys)
        np.save('data/xzz_500_{}'.format(eps), adv_x)
        np.save('data/yzz_500_{}'.format(eps), adv_y.reshape((adv_y.shape[0]*adv_y.shape[1],)))


if __name__ == "__main__":
    flags.DEFINE_integer("nb_epochs", 8, "Number of epochs.")
    flags.DEFINE_float("eps", 0.3, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool(
        "adv_train", False, "Use adversarial training (on PGD adversarial examples)."
    )
    app.run(main)
