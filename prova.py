a = np.random.rand(27)
a = np.reshape(3, 3, 3)

for i in a.shape[0]:
    for y in a.shape[1]:
        a[i, y, :] =