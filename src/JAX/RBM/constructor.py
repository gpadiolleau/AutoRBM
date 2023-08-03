import numpy as np

from typing import Optional, Type, List, Union

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrnd
import jax

from .constructor_helpers import _analyze_graph, _create_instance_parameters, _create_instance_functions

class RBM(object):
    def __init__(self, graph_description, 
                 dtype: Optional[jnp.dtype] = jnp.float32):
        
        self.graph_description = graph_description
        self.dtype = dtype
        self.max_act = 1000.

        _analyze_graph(self)

        _create_instance_parameters(self)
        _create_instance_functions(self)


    def sigmoid_activation(self, x):
        return jnn.sigmoid(x)

    def linear_activation(self, x):
        return x
    
    def softplus_activation(self, x):
        return jnn.softplus(x)
    
    