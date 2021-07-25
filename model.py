import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from easydict import EasyDict
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' #This is a tmp fix of cudnn error
class CNNModel(Model):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.model = tf.keras.models.Sequential([
            Conv2D(32, 3, strides=(1, 1), activation="relu", padding="same"),
            Conv2D(32, 3, strides=(1, 1), activation="relu", padding="same"),
            MaxPool2D((2, 2), strides=(2, 2), padding='same'),
            Conv2D(64, 3, strides=(1, 1), activation="relu", padding="same"),
            Conv2D(64, 3, strides=(1, 1), activation="relu", padding="same"),
            MaxPool2D((2, 2), strides=(2, 2), padding='same'),
            Dropout(0.25),
            Flatten(),
            Dense(256),
            Flatten(),
            Dense(128),
            Dense(10),
        ])

    def call(self, x):
        return self.model(x)

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


def retrieval_model(path, nb_epochs, retrain=False):
    loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    if not retrain:
        try:
            loaded_model = tf.keras.models.load_model(path)
            print("Loaded Model At {}!".format(path))
            loaded_model.compile(optimizer, loss_object, metrics=[tf.keras.metrics.sparse_categorical_accuracy])
            return loaded_model
        except IOError:
            print("Model Not Loaded!")
    print("Training Model From Scratch!")
    # Load training and test data
    data = ld_mnist()
    model = CNNModel()


    # Metrics to track the different accuracies.
    train_loss = tf.metrics.Mean(name="train_loss")
    test_acc_clean = tf.metrics.SparseCategoricalAccuracy()

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_object(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)

    # Train model with adversarial training
    for epoch in range(nb_epochs):
        # keras like display of progress
        progress_bar_train = tf.keras.utils.Progbar(60000)
        for (x, y) in data.train:
            train_step(x, y)
            progress_bar_train.add(x.shape[0], values=[("loss", train_loss.result())])

    # Evaluate on clean and adversarial data
    progress_bar_test = tf.keras.utils.Progbar(10000)
    for x, y in data.test:
        y_pred = model(x)
        test_acc_clean(y, y_pred)
        progress_bar_test.add(x.shape[0])
    model.save(path)
    print(
        "test acc on clean examples (%): {:.3f}".format(test_acc_clean.result() * 100)
    )
    return model


if __name__ == "__main__":
    retrieval_model('models/cnn_model', 50, True)

