# GO0909-3DropoutPadrãoEmDeepLearning
mask = (np.random.rand(*a.shape) < keep_prob) / keep_prob
a_dropout = a * mask
