import numpy as np
from NeuralNetwork import MLP
from Layer import FC
from Activation import Sigmoid


def main():
    # rng = np.random.default_rng()

    # data
    n_features = 4
    X = np.array([[1.0], 
                  [0.5], 
                  [0.2], 
                  [0.8]]) 

    # model
    model = MLP([FC(n_features,1),
                 Sigmoid()
                 ])
    logits = model.forward_pass(X)

    return logits

output = main()
print(output)
