import numpy as np
from scipy.ndimage import gaussian_filter
from tensorflow.keras.losses import Loss

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
    w = ((1 - lambdaa)*p_hat + lambdaa/Q)**-1

    # Normalize w so that expected value is 1
    norm = 0
    for q in range(Q):
        norm += p_hat[q]*w[q]

    w = w/norm

    return w[q_star]


def L_cl(y_true, y_pred):

    Z = y_true
    Z_hat = y_pred

    sum2 = 0

    for h in range(Z.shape[0]):
        for w in range(Z.shape[1]):
            sum3 = 0
            for q in range(Z.shape[2]):
                sum3 += Z[h,w,q] * np.log(Z_hat[h,w,q])
            sum2 += v(Z_h_w=Z[h, w, :]) * sum3
    return -1*sum2


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

# print(f'a: {a}', '\n')
# print(f'b: {b}', '\n')
# print(L_cl(Z_hat=a, Z=a))
# print(L_cl(Z_hat=b, Z=a))
