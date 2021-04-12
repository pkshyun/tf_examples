import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import os

def display_chest_xrays(pos_images, neg_images, k=8):
    nrows = int(k/2)
    ncols = int(k/2)
  
    fig = plt.gcf()
    fig.set_size_inches(nrows * 4, ncols * 4)

    POS_files = [ image_file for image_file in random.sample(pos_images, k=nrows) ]
    NEG_files = [ image_file for image_file in random.sample(neg_images, k=nrows) ]

    for i, image_path in enumerate(POS_files + NEG_files):
        sp = plt.subplot(nrows, ncols, i+1)
        
        if i < nrows:
            sp.set_title('Condition: NORMAL')
        else:
            sp.set_title('Condition: PNEUMONIA')

        img = mpimg.imread(image_path)
        plt.imshow(img, cmap='gray')
    
    plt.show()


def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):

    def weighted_loss(y_true, y_pred):

        loss = 0.0
        for i in range(len(pos_weights)):
            loss += - pos_weights[i] * K.mean(y_true[:,i] * K.log(y_pred[:,i] + epsilon)) \
            - neg_weights[i] * K.mean((1-y_true[:,i]) * K.log(1-y_pred[:,i] + epsilon))
        return loss
    
    return weighted_loss


def compute_class_freqs(labels):

    N = labels.shape[0]
    
    positive_frequencies = np.sum(labels, axis=0)/N
    negative_frequencies = (N - np.sum(labels, axis=0))/N

    return positive_frequencies, negative_frequencies