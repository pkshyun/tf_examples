import tensorflow as tf
import matplotlib.pyplot as plt
import random

def plot_random_images(images, labels, shape):
    
    ncols = 10
    n_images = len(images)
    random_indices = random.choices(range(n_images), k=ncols)

    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, 4)

    for i in range(ncols):
        index = random_indices[i]
        sp = plt.subplot(1, ncols, i+1)
        sp.axis('off')

        plt.title(labels[index], fontsize=20)
        plt.imshow(images[index].reshape(shape))


def dataset_to_numpy(dataset, augmented=False):

    if augmented:
        for digits, (label_vec, b_box) in dataset.batch(32):
            X = digits.numpy()
            labels = label_vec.numpy()
            bboxes = b_box.numpy()

            return (X, labels, bboxes)

    for digits, classes in dataset.batch(32):
        X = digits.numpy()
        y = classes.numpy()
        
        return (X, y)


def place_images_on_canvas(image, label):

    x_min = tf.random.uniform((), 0, 75 - 28 + 1, dtype=tf.int32) # scalar: 0 - 47
    y_min = tf.random.uniform((), 0, 75 - 28 + 1, dtype=tf.int32) # scalar: 0 - 47

    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.pad_to_bounding_box(image, x_min, y_min, 75, 75)

    # normalize
    x_min = tf.cast(x_min, tf.float32)
    y_min = tf.cast(y_min, tf.float32)
    x_min /= 75.
    y_min /= 75.
    x_max = (x_min + 28) / 75.
    y_max = (y_min + 28) /  75.

    b_box = [x_min, y_min, x_max, y_max]
    label_vec = tf.one_hot(label, 10)

    return image, (label_vec, b_box)