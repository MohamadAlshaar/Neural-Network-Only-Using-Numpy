import numpy as np

class ReLu:   # f(z)=0 if z <=0, 1 otherwise
    def __call__(self, pre_activated_output):
        return np.where(pre_activated_output <= 0, 0, 1) * pre_activated_output
        


class Sigmoid:    # f(x)= 1/(1+e^(-z))
    def __call__(self, pre_activated_output):
        return 1 / (1 + np.exp(-pre_activated_output))
    


class Softmax:  # e^[e^z_n] / e^z_n 
    def __call__(self, pre_activated_output):
        exp_diff = np.exp(pre_activated_output - np.max(pre_activated_output, axis=1, keepdims=True))
        nominator = np.sum(exp_diff)
        return exp_diff/nominator

        