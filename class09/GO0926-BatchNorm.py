# GO0926-BatchNorm
class BatchNorm:
    def __init__(self, num_features, momentum=0.9):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.momentum = momentum

    def forward(self, x, training=True):
        if training:
            # Estatísticas do batch
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            # Normalizar
            x_norm = (x - mean) / np.sqrt(var + 1e-5)
            # Atualizar running stats
            self.running_mean = (self.momentum * self.running_mean
                                + (1-self.momentum) * mean)
            self.running_var = (self.momentum * self.running_var 
                               + (1-self.momentum) * var)
        else:
            # Usar running stats
            x_norm = ((x - self.running_mean) /
                     np.sqrt(self.running_var + 1e-5))
        # Escalar e deslocar
        out = self.gamma * x_norm + self.beta
        return out
