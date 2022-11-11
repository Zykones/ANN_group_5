import numpy as np
from numpy import random
import math
import matplotlib.pyplot as plt


### Task 2
class Layer:
    def __init__(self, n_units, input_units):
        self.n_units = n_units
        self.input_units = input_units

        #generate bias and weight
        self.bias = np.zeros(n_units)
        self.weight = random.normal(size=(input_units,n_units))

        self.layer_input = np.empty(shape=(1,input_units))
        self.layer_preactivation = np.empty(shape=(1,n_units))
        self.layer_activation = np.empty(shape=(1,n_units))


    def forward_step(self, input):
        self.layer_input = input
        preactivation = input @ self.weight + self.bias    #@=Matrix Multiplication
        self.layer_preactivation = preactivation

        activation = np.maximum(0, preactivation)
        self.layer_activation = activation

        return activation

    def backward_step(self, dl_activation, eta):

        #                      |       dl_activation                       
        # input_layer-> hidden_layer -> out_layer -> Loss 

        sig = np.vectorize(self.sigmoid)

        #Got this from a Tutor: seems to be wrong...
        #gradient_weight = np.array([np.where(self.layer_preactivation>0, [1.0], [0.0]) * dl_activation])

        gradient_weight = np.array([(sig(self.layer_preactivation) * (1- sig(self.layer_preactivation))) * dl_activation])

        #gradient_bias = np.where(self.layer_preactivation>0,[1.],[0.]) * dl_activation
        gradient_bias = (sig(self.layer_preactivation) * (1- sig(self.layer_preactivation))) * dl_activation

        #gradient_input = (np.where(self.layer_preactivation>0,[1.],[0.]) * dl_activation) @ np.transpose(self.weight)
        gradient_input = (sig(self.layer_preactivation) * (1- sig(self.layer_preactivation))) @ np.transpose(self.weight)

        self.weight = self.weight - (gradient_weight * eta)
        self.bias = self.bias - (gradient_bias * eta)

        return gradient_input
    
    def sigmoid(self, x):
        return 1 / (1+ math.exp(-x))

### Task 3
class MLP:           # layer list = [1,10,1]
    def __init__(self, input_list, eta:float):

        self.learning_rate = eta
        
        self.input_list = input_list
        
        self.layer_list = []

        #input Layer Anzalh neuronen = Anzahl input Werte
        self.layer_list.append(Layer(self.input_list[0], self.input_list[0]))

        for i in range(1, len(input_list)):
            self.layer_list.append(Layer(input_list[i],input_list[i-1]))

    def forward_step(self, input:np.ndarray):
        # input = Array with values to pass
        
        last_activation = self.layer_list[0].forward_step(input)

        for i in self.layer_list[1:]:
            last_activation = i.forward_step(last_activation)
        
        return last_activation

    def backpropagation(self, loss:float):
        # loss = scalar value 

        loss_array = np.array(loss, ndmin=1) # Change Axis 

        last_backpropagation = self.layer_list[-1].backward_step(loss_array, self.learning_rate)

        for i in reversed(self.layer_list[:-1]):
            last_backpropagation = i.backward_step(last_backpropagation, self.learning_rate)



# Task 4
if __name__ == "__main__":

    ### Task 1
    #Learning Rate
    eta = 0.1
    epochs = 1000
    layer_list = [1,10,1]
    training_size = 100

    x = x=random.uniform(0, 1, size=training_size)

    t = [i**3-i**2+1 for i in x]
    t = np.array(t)

    neural_network = MLP(layer_list, eta)

    def loss_funktion(target, output):
        return 0.5 * (output - target[0])**2 

    def train(nn:MLP, training_data:np.ndarray, target_data:np.ndarray, eta:float, epochs:int):
        
        loss_list = np.array([], dtype=float)

        for i in range(epochs):

            for j in range(0,len(training_data)):
                    #cast to numpy for trtain and targets
                np_train = np.array(training_data[j], ndmin=1)
                #np_train = np.reshape(np_train, (1,1))
                nn_output = nn.forward_step(np_train)

                np_target = np.array(target_data[j],ndmin=1)
                #np_target = np.reshape(np_target, (1,1))
                loss_list = np.append(loss_list, loss_funktion(np_target, nn_output[0].astype(float)))

                nn.backpropagation(loss_list[-1])

            if(i%50==0):
                print("Loss: {0}".format(loss_list[-1].astype(float)))
        
        return loss_list
                
    plot_x = np.arange(epochs*training_size)
    loss = train(neural_network, x, t, eta, epochs)
    ### Task 2.5
    fig, ax = plt.subplots()

    ax.plot(plot_x, loss)

    plt.show()


