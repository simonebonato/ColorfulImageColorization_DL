import numpy as np
import sklearn.neighbors as nn

# Load the array of quantized ab values
q_ab = np.load("pts_in_hull.npy")
nb_q = q_ab.shape[0]
# Fit a NN to q_ab
nn_finder = nn.NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(q_ab)


def soft_encoding(image_ab, nn_finder, nb_q):
    Z_list = [None]*image_ab.shape[0]

    for n in range(image_ab.shape[0]):
        h, w = image_ab[n].shape[:2]
        # print(f'image_ab shape: {image_ab.shape}')
        img_a = image_ab[n][:, :, 0]
        img_b = image_ab[n][:, :, 1]

        a = np.ravel(img_a)
        b = np.ravel(img_b)
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
        Z_list[n] = Z.reshape((h, w, nb_q))

    return np.array(Z_list)


def v(Z):
    # Uniform distribution parameter
    lambdaa = 0.5

    # Gaussian Kernel width
    sigma = 5

    Q = 313
    q_star = np.argmax(Z, axis=-1)

    # Estimated probability distribution for colors
    p = Z
    # print(p.shape)
    # Smoothen distribution of estimated probability distribution for colors
    p_hat = np.exp(-p ** 2 / (2 * sigma ** 2))
    w = ((1 - lambdaa) * p_hat + lambdaa / Q) ** -1

    # Normalize w so that expected value is 1
    norm = np.sum(p_hat[:, :, :] * w[:, :, :], axis=-1)
    norm = norm.reshape(Z.shape[0], Z.shape[1], Z.shape[2], 1)
    norm = np.tile(norm, (1, 1, 1, Z.shape[-1]))
    w = w / norm
    q_star = np.reshape(q_star, newshape=(q_star.shape[0], q_star.shape[1], q_star.shape[2], 1))
    return np.take_along_axis(w, q_star, axis=-1)


def L_cl(y_true, y_pred):
    """
    y_true: batch_size x 64 x 64 x 2 from Generator
    y_pred: batch_size x 64 x 64 x 313 from CNN
    """

    """TODO: make sure this works for batches of data"""
    batch_size = y_true.shape[0]
    loss = 0

    # takes y_true[n] and returns b_size x 64 x 64 x 313 soft encoded version
    Z = soft_encoding(image_ab=y_true, nn_finder=nn_finder, nb_q=nb_q)
    Z_hat = y_pred
    # print(Z.shape)

    # class re-balancing for Z, returns 64 x 64 x 1 to rebal. the weights
    # one probability value for each pixel
    v_Z = v(Z)
    v_Z = v_Z.reshape(Z.shape[0], Z.shape[1], Z.shape[2])
    v_Z *= (Z * np.log(Z_hat)).sum(axis = -1)

    loss += np.sum(v_Z)
    # print(loss)
    return -1 * loss/batch_size

