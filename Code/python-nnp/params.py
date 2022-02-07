from itertools import product

LEARNING_RATES = [0.01, 0.05, 0.1, 0.2]
ITERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
PROJECTIONS = ['LMDS', 'TSNE', 'PCA']
PARAM_GRID = list(product(PROJECTIONS, ITERS, LEARNING_RATES))
