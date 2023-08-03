# AutoRBM
AutoRBM automatically creates a RBM object that follows the given graph description. It dynamically creates object functions, functions names, functions docstrings and functions signatures according to the graph description.
This results in a fully operational RBM architecture with functions inputs & functions names named following the layers names given in the graph description.

AutoRBM is able to :
 - build Factored RBM (gated factored RBM only, no gated RBM)
 - build Deep RBM (multi-hidden layer)
 - deal with gaussian and binary layers (Relu layers are coming)
 - automatically detect conditional layers
 - automatically creates parameters
 - build probabilities and samples functions wrt to needed inputs (first order and second order inputs if there are message-passing/linear layer)
 - perform a network stabilisation using Gibbs sampling (for deep propagation through the network and for co-dependant layers)
 - CD-k gradients estimation

This project use JAX as backend but it can easily be adapted to other ML libraries or Numpy

I added a notebook ```test_autoRBM``` you can play with to check how it works.

Feel free to take this code to play with and thanks for any help you could give me :) 
