import numpy as np

class layer:
    def __init__(self, number_inputs, number_outputs, alpha, batch_size, bias= False, activation_function=None):
         self.number_inputs = number_inputs
         self.number_outputs = number_outputs
         self.number_alpha = alpha
         self.batch_size = batch_size
         self.activation_function = activation_function
         self.bias = bias
         self.weights = np.random.rand(self.number_inputs, self.number_outputs)

         if bias:
            self.weights = np.vstack((self.weights, np.random.rand(1, self.number_outputs)))

    def get_batch_size(self):
        return self.batch_size
    
    def set_alpha(self, update_alpha):
        self.alpha = update_alpha
         
      

class all_layers:
    def __init__(self, *layers):
        self.model= list(layers)
    
    def append(self, *layers):
        for layer in layers:
            self.model.append(layer)

    def set_alpha(self, update_alpha):
        for layer in self.model:
            layer.set_alpha(update_alpha)
