import numpy as np


def prob_to_point_est(Z, T=0.38):
    q_ab = np.load("pts_in_hull.npy")

    # Z is a vector with dims [H, W, Q=313]
    # Each Q is a probability that the pixel has a specific gamut color
    new_p = np.zeros_like(Z)
    image_out = np.zeros(shape=(Z.shape[0], Z.shape[1], 2))

    for h in range(Z.shape[0]):
        for w in range(Z.shape[1]):
            probs = Z[h, w, :]
            new_p[h, w, :] = np.exp(np.log(probs) / T) / np.sum(np.exp(np.log(probs) / T))
            ab = (new_p[h, w, :][:, np.newaxis] * q_ab).sum(axis=0)
            image_out[h, w, :] = ab

    # Returns HxWx2 (Y)
    return image_out


# q_ab = np.load("pts_in_hull.npy")
# nb_q = q_ab.shape[0]
#
# probs = np.random.rand(256, 256, 313)
# for i in range(probs.shape[0]):
#     for y in range(probs.shape[1]):
#         probs[i, y, :] /= np.sum(probs[i, y, :])
#
# image_out = prob_to_point_est(probs)
