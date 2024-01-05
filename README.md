A ground-up implementation of an autograd engine:

  - The computational DAG is implicitly registered during the forward pass.
  - Each time a node is constructed via an operation (e.g. +-/*, pow, exp, relu, etc...) the corresponding backward function is registered on the result node.
  - The DAG is explicitly constructed from the loss node via a DFS, and the gradients are propagated through the graph by calling .backward() on this node.
  - The optimiser then steps through the model's parameters updating the weights (currently only Adam is implemented).
