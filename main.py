import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from easydict import EasyDict
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D

from model import CNNModel, retrieval_model
import os
from defenses import rpgd_defense, zigzag_defense

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' #This is a tmp fix of cudnn error
FLAGS = flags.FLAGS

def ld_mnist():
    """Load training and test data."""

    def convert_types(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    dataset, info = tfds.load(
        "mnist", data_dir="gs://tfds-data/datasets", with_info=True, as_supervised=True
    )
    mnist_train, mnist_test = dataset["train"], dataset["test"]
    mnist_train = mnist_train.map(convert_types).shuffle(10000).batch(128)
    mnist_test = mnist_test.map(convert_types).batch(128)
    return EasyDict(train=mnist_train, test=mnist_test)


def main(_):
    for eps_iter in [0.005,]:
        for nb in [300, 400, 500, 600, 700, 800, 900, 1000]:
            for rand_init in [1]:
                # Load training and test data
                test_acc_clean = tf.metrics.SparseCategoricalAccuracy()
                test_acc_fgsm = tf.metrics.SparseCategoricalAccuracy()
                test_acc_pgd = tf.metrics.SparseCategoricalAccuracy()

                test_acc_aae = tf.metrics.SparseCategoricalAccuracy()
                test_acc_fgsm_aae = tf.metrics.SparseCategoricalAccuracy()
                test_acc_pgd_aae = tf.metrics.SparseCategoricalAccuracy()
                data = ld_mnist()
                model = retrieval_model('models/cnn_model', 50, False)

                i = 0
                for x, y in data.test:
                    print(i)
                    i = i+1
                    # print(y)
                    # y_pred = model(x)
                    # test_acc_clean(y, y_pred)
                    #
                    # # x_fgm = fast_gradient_method(model, x, FLAGS.eps, np.inf)
                    # # y_pred_fgm = model(x_fgm)
                    # # test_acc_fgsm(y, y_pred_fgm)
                    #
                    # x_pgd = projected_gradient_descent(model, x, 0.2, 0.01, 100, np.inf, y=y, rand_init=0.2)
                    # # x_pgd = fast_gradient_method(model, x, 0.2, np.inf)
                    # y_pred_pgd = model(x_pgd)
                    # test_acc_pgd(y, y_pred_pgd)
                    #
                    # x_aae = zigzag_defense(model, x, 0.2, eps_iter, nb, eps_iter, np.inf, y=y, rand_init=rand_init * eps_iter)
                    # # x_aae = rpgd_defense(model, x, 0.2, 0.01, 40, np.inf)
                    # y_pred_aae = model(x_aae)
                    # test_acc_aae(y, y_pred_aae)
                    #
                    # # x_fgm_aae = fast_gradient_method(model, x_aae, FLAGS.eps, np.inf)
                    # # y_pred_fgm_aae = model(x_fgm_aae)
                    # # test_acc_fgsm_aae(y, y_pred_fgm_aae)
                    #
                    #
                    # x_pgd_aae = projected_gradient_descent(model, x_aae, 0.2, 0.01, 100, np.inf, y=y, rand_init=0.2)
                    # # x_pgd_aae = fast_gradient_method(model, x_aae, 0.2, np.inf)
                    # y_pred_pgd_aae = model(x_pgd_aae)
                    # test_acc_pgd_aae(y, y_pred_pgd_aae)

                    # i+=1
                    # if(i > 3):
                    #     break
                print('EPS iter: {}, nb: {}, rand_int: {}'.format(eps_iter, nb, rand_init*eps_iter))
                # print(
                #     "test acc on FGM adversarial examples (%): {:.3f}".format(
                #         test_acc_fgsm.result() * 100
                #     )
                # )
                print(
                    "test acc on PGD adversarial examples (%): {:.3f}".format(
                        test_acc_pgd.result() * 100
                    )
                )
                print(
                    "test acc Of AAE (%): {:.3f}".format(
                        test_acc_aae.result() * 100
                    )
                )
                print(
                    "test acc Of AAE on PGD adversarial examples (%): {:.3f}".format(
                        test_acc_pgd_aae.result() * 100
                    )
                )
if __name__ == "__main__":
    flags.DEFINE_float("eps", 0.3, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool(
        "adv_train", False, "Use adversarial training (on PGD adversarial examples)."
    )
    app.run(main)
