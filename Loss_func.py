import numpy as np
import sklearn.neighbors as nn


def soft_encoding(image_ab, nn_finder, nb_q):
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

    # format the target
    Z = np.zeros((ab.shape[0], nb_q))
    idx_pts = np.arange(ab.shape[0])[:, np.newaxis]
    Z[idx_pts, idx_neigh] = wts
    Z = Z.reshape((h, w, nb_q))

    return Z


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
    p_hat = np.exp(-p ** 2 / (2 * sigma ** 2))
    w = ((1 - lambdaa) * p_hat + lambdaa / Q) ** -1

    # Normalize w so that expected value is 1
    norm = 0
    for q in range(Q):
        norm += p_hat[q] * w[q]

    w = w / norm

    return w[q_star]


def L_cl(y_true, y_pred):
    # y_true is HxWx2 (ab without L component of image), y_pred is HxWxQ (predicted prob distribution)

    # Load the array of quantized ab value
    q_ab = np.load("pts_in_hull.npy")
    nb_q = q_ab.shape[0]

    # Fit a NN to q_ab
    nn_finder = nn.NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(q_ab)

    Z = soft_encoding(image_ab=y_true, nn_finder=nn_finder, nb_q=nb_q)
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
# a = a.reshape(3,3,3)
# for i in range(a.shape[0]):
#     for y in range(a.shape[1]):
#         a[i, y, :] /= np.sum(a[i, y, :])
#
# b = np.random.rand(27)
# b = b.reshape(3,3,3)
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


def prob_to_point_est(Z, temperature=0.38):
    # Z is a vector with dims [H, W, Q=313]
    # each Q is a probability that the pixel has a specific gamut color
    new_p = np.copy(Z)
    colors = np.zeros(shape=(Z.shape[0], Z.shape[1], 1))
    for h in range(Z.shape[0]):
        for w in range(Z.shape[1]):
            probs = Z[h, w, :]
            new_p[h, w, :] = np.exp(np.log(probs) / temperature) / \
                             np.sum(np.exp(np.log(probs) / temperature))
            colors[h, w] = np.argmax(new_p[h, w, :])

    return new_p, colors


# q_ab = np.load("data/pts_in_hull.npy")
# nb_q = q_ab.shape[0]
# print(q_ab)
# print(nb_q)

probs = np.random.rand(256, 256, 313)
for i in tqdm.tqdm(range(probs.shape[0])):
    for y in range(probs.shape[1]):
        probs[i, y, :] /= np.sum(probs[i, y, :])

new_p, colors = prob_to_point_est(probs)
print(new_p)
print(colors[0, 0])
