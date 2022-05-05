import numpy as np
from scipy.ndimage import gaussian_filter


def v(Z_h_w):
    # Uniform distribution parameter
    lambdaa = 0.5

    # Gaussian Kernel width
    sigma = 5

    Q = 313
    q_star = np.argmax(Z_h_w)

    # Estimated probability distribution for colors
    p = Z_h_w

    # Smoothen distribution of estimated probability distribution for colors
    p_hat = gaussian_filter(input=p, sigma=sigma)
    w = ((1 - lambdaa) * p_hat + lambdaa / Q) ** -1

    # Normalize w so that expected value is 1
    norm = 0
    for q in range(Q):
        norm += p_hat[q] * w[q]

    w = w / norm

    return w[q_star]


def L_cl(y_true, y_pred):
    Z = y_true
    Z_hat = y_pred

    sum2 = 0

    for h in range(Z.shape[0]):
        for w in range(Z.shape[1]):
            sum3 = 0
            for q in range(Z.shape[2]):
                sum3 += Z[h, w, q] * np.log(Z_hat[h, w, q])
            sum2 += v(Z_h_w=Z[h, w, :]) * sum3
    return -1 * sum2


# a = np.random.rand(27)
# a = a.reshape((3, 3, 3))
# for i in range(a.shape[0]):
#     for y in range(a.shape[1]):
#         a[i, y, :] /= np.sum(a[i, y, :])
#
# b = np.random.rand(27)
# b = b.reshape((3, 3, 3))
# for i in range(b.shape[0]):
#     for y in range(b.shape[1]):
#         b[i, y, :] /= np.sum(b[i, y, :])
#
# # print(f'a: {a}', '\n')
# # print(f'b: {b}', '\n')
# print(L_cl(y_true=a, y_pred=a))
# print(L_cl(y_true=b, y_pred=a))


def get_soft_encoding(image_ab, nn_finder, nb_q):
    h, w = image_ab.shape[:2]
    a = np.ravel(image_ab[:, :, 0])
    b = np.ravel(image_ab[:, :, 1])
    ab = np.vstack((a, b)).T
    # Get the distance to and the idx of the nearest neighbors
    dist_neighb, idx_neigh = nn_finder.kneighbors(ab)
    # Smooth the weights with a gaussian kernel
    sigma_neighbor = 5
    wts = np.exp(-dist_neighb ** 2 / (2 * sigma_neighbor ** 2))
    wts = wts / np.sum(wts, axis=1)[:, np.newaxis]
    # format the tar get
    y = np.zeros((ab.shape[0], nb_q))
    idx_pts = np.arange(ab.shape[0])[:, np.newaxis]
    y[idx_pts, idx_neigh] = wts
    y = y.reshape(h, w, nb_q)
    return y

q_ab = np.load("data/pts_in_hull.npy")
nb_q = q_ab.shape[0]
print(q_ab)
print(nb_q)