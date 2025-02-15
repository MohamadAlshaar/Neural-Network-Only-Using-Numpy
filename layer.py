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
            self.weights = np.vstack((self.weights, np.random.rand(1, self.number_outputs)))  # match the bias by making the matrix vertical, 
                                                                                                #think of the bias as an input neuron that we add to the rest of the neurons 

    def get_batch_size(self):
        return self.batch_size
    
    def set_alpha(self, update_alpha):
        self.alpha = update_alpha
         
      

class all_layers:
    def __init__(self, *layers):
        self.model= list(layers)
    
    def append(self, *layers):     #so we can add more layers if we wish to
        for layer in layers:
            self.model.append(layer)

    def set_alpha(self, update_alpha):   #update alhpa value for all layers
        for layer in self.model:
            layer.set_alpha(update_alpha)
