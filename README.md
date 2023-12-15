First iteration of network passing all tests!

However, the network is impressively slow. The following updates will address several issues:

  1. New Value objects are currently being created with each operation. Instead, register gradients and computational graph once during instantiation of the model (can pass a dummy input and register ops at each step. Cache the computational graph to use for future backprop passes).
  2. Since the graph is static, once the graph has been registered I can operate on the data stored inside the Value objects. Inside the model this can be done by created a weights matrix. The output Values' data can be changed without creating new objects.
  3. Keep track and store Values during the loss calculation for the same reason - don't want to recalculate the values as this will change the pointers stored in the static graph (making it non-static and defeating the purpose of calculating it once)
  4. Get rid of the proxy class - this takes up far too much room and is not necessary. 
