A ground-up implementation of an autograd engine, demonstrated on dummy data as well as MNIST.

During the forward pass, node operations and backward hooks are registered.
The computational graph is constructed from the loss node via DFS and the partial derivative of the loss w.r.t. the model's 
parameters is automatically propagated through the graph. The optimiser then updates the model weights.
