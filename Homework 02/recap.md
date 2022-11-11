# Recap
1 - The activation function determines the offset with which the neuron is activated or which value it should output. The offset does not have to be linear, it can follow any other differentiable function.

2 - The sigmoid function maps all input values to a range between 0 and 1, with a smooth, differentiable transition between the values 0 and 1 between -4 and 4. 
    The Relu function is linear for x >= 0 and 0 otherwise.

3 - Sigmoid: sig'(x) = sig(x) * (1-sig(x))
    Relu: relu' = 1 for x >= 0, else 0

4 - In application it has been shown that networks with relu neurons converge faster. Sigmoid is the differentiable version of the step function.
