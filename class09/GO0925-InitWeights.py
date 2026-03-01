# GO0925-InitWeights
def init_weights(n_in, n_out, activation='relu'):
    if activation == 'relu':
        # He initialization
        std = np.sqrt(2.0 / n_in)
    elif activation in ['sigmoid', 'tanh']:
        # Xavier initialization
        std = np.sqrt(2.0 / (n_in + n_out))
    else:
        std = 0.01

    W = np.random.randn(n_in, n_out) * std
    b = np.zeros((1, n_out))
    return W, b
