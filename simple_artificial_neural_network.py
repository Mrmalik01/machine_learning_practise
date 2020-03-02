import numpy as np
from pprint import PrettyPrinter

class ArtificialNeuralNetwork:
    
    def __init__(self, n_inputs, n_outputs=1, inputs=[]):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.layers = []
        self.inputs = inputs
        
    def addInputs(self, inputs):
        if len(inputs) == self.n_inputs:
            self.inputs = inputs
        
    def addHiddenLayer(self, nodes):
        layer = {}
        if self.layers == []:
            previous_nodes = self.n_inputs
        else:
            previous_nodes = len(self.layers[-2])
        #previous_nodes = self.n_inputs if self.layers == [] else len(self.layers[-2])
        for node in range(nodes):
                layer['node_{}'.format(node+1)] = {
                    'weights' : np.around(np.random.uniform(size=previous_nodes), decimals=4), 
                    'bias' : np.around(np.random.uniform(size=1), decimals=4)
        }       
        if len(self.layers) <= 1:
            self.layers.append(layer)
        else:
            self.layers[-1] = layer
        self._computeOutputLayer()
        
    def _computeOutputLayer(self):
        if self.layers == []:
            previous_nodes = self.n_inputs
        else:
            previous_nodes = len(self.layers[-1])
        #previous_nodes = self.n_inputs if len(self.layers) == 0 else len(self.layers[-2])
        layer = {}
        for node in range(self.n_outputs):
                layer['node_{}'.format(node+1)] = {
                    "weights" : np.around(np.random.uniform(size=previous_nodes), decimals=4),
                    "bias" : np.around(np.random.uniform(size=1), decimals=4)
                }
        self.layers.append(layer)
            
    def showDefinition(self):
        result = {
            "input_layer" : {
                "nodes" : self.inputs if self.inputs != [] else self.n_inputs
            }
        }
        for index, layer in enumerate(self.layers):
            result['layer_{}'.format(index+1) if index < len(self.layers)-1 else  "output_layer"] = self.layers[index]
        return result
    
    def _compute_weighted_sum(self, weights, inputs, bias):
        return np.sum(inputs * weights) + bias
    
    def _compute_activation_function(self, a):
        return 1.0 / (1.0 + np.exp(-a))
    
    def forwardPropagation(self):
        inputs = self.inputs
        for index, layer in enumerate(self.layers):
            weighted_sum_each_node = [np.around(self._compute_weighted_sum(layer[node]['weights'], inputs, layer[node]['bias']), decimals=4) for node in layer]
            inputs = [self._compute_activation_function(weighted_sum) for weighted_sum in weighted_sum_each_node]
        return inputs

            
    
    
network = ArtificialNeuralNetwork(5, 1, np.around(np.random.uniform(size=5), decimals=4))
network.addHiddenLayer(2)
network.addHiddenLayer(4)
pp = PrettyPrinter(indent = 3)
pp.pprint(network.showDefinition())
print("Result : {}".format(network.forwardPropagation()))