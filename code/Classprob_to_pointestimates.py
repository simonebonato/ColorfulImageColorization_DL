import numpy as np
import cv2


def prob_to_point_est(Z, T=0.38):
    q_ab = np.load("pts_in_hull.npy")

    # Z is a vector with dims [64, 64, Q=313]
    # Each Q is a probability that the pixel has a specific gamut color
    new_p = np.zeros_like(Z)
    image_out = np.zeros(shape=(Z.shape[0], Z.shape[1], 2))

    for h in range(Z.shape[0]):
        for w in range(Z.shape[1]):
            probs = Z[h, w, :]
            new_p[h, w, :] = np.exp(np.log(probs) / T) / np.sum(np.exp(np.log(probs) / T))
            ab = (new_p[h, w, :][:, np.newaxis] * q_ab).sum(axis=0)
            image_out[h, w, :] = ab

    # Returns 64x64x2 (Y)
    return image_out


def reconstruct_image(X, y_pred):
    """
    X: CNN input - L channel [batch_size x H x W x 1]
    y: CNN output - ab channels [batch_size x 64 x 64 x 2]
    return: colored version of the image in LAB
    """
    batch_size = X.shape[0]
    h, w = X.shape[1:3]
    output_imgs = np.zeros(shape=(batch_size, h, w, 3))
    for i in range(batch_size):
        y = prob_to_point_est(y_pred[i])
        ab_resized = cv2.resize(y, (h, w), cv2.INTER_CUBIC)
        output_imgs[i, :, :, 0] = np.squeeze(X[i])
        output_imgs[i, :, :, 1:] = ab_resized
    return output_imgs


def reconstruct_gt_image(X, y_true):
    """
    X: CNN input - L channel [batch_size x H x W x 1]
    y: CNN output - ab channels [batch_size x 64 x 64 x 2]
    return: colored version of the image in LAB
    """
    batch_size = X.shape[0]
    h, w = X.shape[1:3]
    output_imgs = np.zeros(shape=(batch_size, h, w, 3))
    for i in range(batch_size):
        y = y_true[i]
        ab_resized = cv2.resize(y, (h, w), cv2.INTER_CUBIC)
        output_imgs[i, :, :, 0] = np.squeeze(X[i])
        output_imgs[i, :, :, 1:] = ab_resized
    return output_imgs


# q_ab = np.load("pts_in_hull.npy")
# nb_q = q_ab.shape[0]
#
# probs = np.random.rand(256, 256, 313)
# for i in range(probs.shape[0]):
#     for y in range(probs.shape[1]):
#         probs[i, y, :] /= np.sum(probs[i, y, :])
#
# image_out = reconstruct_image(pro)
