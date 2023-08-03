from dataclasses import dataclass, field, asdict
from typing import List, Union, Optional
import warnings

import numpy as np
import jax.numpy as jnp

LAYER_TYPES = {"Binary", "Gaussian", "ReLU"}


@dataclass
class NeuralLayer:
    """
    Represents a neural layer in a graph.

    Args:
        name (str): The name of the layer.
        type (str): The type of the layer. Must be one of {"Binary", "Gaussian", "ReLU"}.
        units_number (int): The number of units in the layer.
        priority (int): Corresponds to the stage of the layer in the graph. Needed for safe bottom-up/top-down propagations in non-trivial graphs.
                        The lowest priority is 0 and corresponds to visible layer.
        initial_biases (optional: float or array): initial value for layer biases
        initial_sigmas (optional: float or array): initial value for layer sigmas, used only for type 'Gaussian'

    Raises:
        ValueError: If `units_number` is less than 1.
        TypeError: If `units_number` is not of type `int`.
        TypeError: If `type` is not one of {"Binary", "Gaussian", "ReLU"}.
    """
    name: str
    type: str
    units_number: int
    priority: int
    initial_biases: Optional[Union[None, float, np.ndarray, jnp.ndarray]] = None
    initial_sigmas: Optional[Union[None, float, np.ndarray, jnp.ndarray]] = None
    

    def __post_init__(self):
        if self.units_number < 1:
            raise ValueError(f"'units_number' shoudl greater or equal to 1, but received {self.units_number}")
        if not isinstance(self.units_number, int):
            raise TypeError(f"'units_number' should be type {int.__name__}, but received {type(self.units_number).__name__}")
        if not (self.type in LAYER_TYPES):
            raise TypeError(f"'type' must be in {LAYER_TYPES}, but received {self.type}")
        if self.initial_biases is not None:
            try:
                assert (isinstance(self.initial_biases,float) or isinstance(self.initial_biases, jnp.ndarray) or isinstance(self.initial_biases, np.ndarray))
            except:
                raise TypeError(f"initial_biases must be in types '{float.__name__}', {np.ndarray.__name__} or JAX {jnp.ndarray.__name__}, but received {type(self.initial_biases)}")
        if self.initial_sigmas is not None:
            try:
                assert (isinstance(self.initial_sigmas,float) or isinstance(self.initial_sigmas, jnp.ndarray) or isinstance(self.initial_sigmas, np.ndarray))
            except:
                raise TypeError(f"initial_sigmas must be in types '{float.__name__}', {np.ndarray.__name__} or JAX {jnp.ndarray.__name__}, but received {type(self.initial_sigmas)}")
        

@dataclass
class LinearLayer:
    """
    Represents a linear layer in a graph.

    Args:
        name (str): The name of the layer.
        units_number (int): The number of units in the layer.

    Raises:
        ValueError: If `units_number` is less than 1.
        TypeError: If `units_number` is not of type `int`.
    """
    name: str
    units_number: int

    def __post_init__(self):
        if self.units_number < 1:
            raise ValueError(f"'units_number' shoudl greater or equal to 1, but received {self.units_number}")
        if not isinstance(self.units_number, int):
            raise TypeError(f"'units_number' should be type {int.__name__}, but received {type(self.units_number).__name__}")

@dataclass
class InteractionLink:
    """
    Represents an interaction link between nodes in a graph.

    Args:
        origin_node (Union[NeuralLayer,LinearLayer]): The origin node.
        target_node (Union[NeuralLayer,LinearLayer]): The target node.
        bidirectional (bool, optional): Whether the link is bidirectional. Default is True.

    Raises:
        None
    """
    origin_node: Union[NeuralLayer,LinearLayer]
    target_node: Union[NeuralLayer,LinearLayer]
    bidirectional: bool = True
    initial_weights: Optional[Union[None, float, np.ndarray, jnp.ndarray]] = None

    def __post_init__(self):
        if not (isinstance(self.origin_node, NeuralLayer) or isinstance(self.origin_node, LinearLayer)):
            raise TypeError(f"Origin node {self.origin_node} must be an instance of NeuralLayer or LinearLayer dataclass")
        if not (isinstance(self.target_node, NeuralLayer) or isinstance(self.target_node, LinearLayer)):
            raise TypeError(f"Target node {self.target_node} must be an instance of NeuralLayer or LinearLayer dataclass")
        if self.initial_weights is not None:
            try:
                assert (isinstance(self.initial_weights,float) or isinstance(self.initial_weights, jnp.ndarray) or isinstance(self.initial_weights, np.ndarray))
            except:
                raise TypeError(f"initial_weights must be in types '{float.__name__}', {np.ndarray.__name__} or JAX {jnp.ndarray.__name__}, but received {type(self.initial_weights)}")
        type = 'bidirectional' if self.bidirectional else 'directional'
        self.name = type + "_link_from_" + self.origin_node.name + "_to_" + self.target_node.name

@dataclass
class GraphDescription:
    """
    Represents a graph description.

    Args:
        nodes (List[Union[NeuralLayer,LinearLayer]], optional): The list of nodes in the graph. Default is an empty list.
        links (List[InteractionLink], optional): The list of links in the graph. Default is an empty list.

    Raises:
        ValueError: If a link is attached to a node that does not appear in the graph description.
        UserWarning: If there are nodes in the graph that are not connected to any link.
    """
    nodes: List[Union[NeuralLayer,LinearLayer]] = field(default_factory=list)
    links: List[InteractionLink] = field(default_factory=list)

    def __post_init__(self):
        for node in self.nodes:
            if not (isinstance(node, NeuralLayer) or isinstance(node, LinearLayer)):
                raise TypeError(f"Node {node} must be an instance of NeuralLayer or LinearLayer dataclass")

        self.layers_names = [layer.name for layer in self.nodes]
        _temp_layers_names = self.layers_names.copy()
        for link in self.links:
            if not isinstance(link, InteractionLink):
                raise TypeError(f"Link {link} must be an instance of Link dataclass")
            if not (link.origin_node.name in self.layers_names and link.target_node.name in self.layers_names):
                raise ValueError(f"A node, the link: '{link.name}' is attached,  appears not to be in the graph description.\n"
                                 f"Please check layers names or links attached nodes.")
            else:
                try: _temp_layers_names.remove(link.origin_node.name)
                except: pass
                try: _temp_layers_names.remove(link.target_node.name)
                except: pass
        if bool(_temp_layers_names):
            warnings.warn(f"Layers nodes {_temp_layers_names} are not connected to any link.")

    def get_description_as_dict(self):
        return asdict(self).copy()
    
    def get_description_as_matrix(self):
        desc_matrix = np.zeros((len(self.layers_names), len(self.layers_names)))

        for link in self.links:
            idx = (self.layers_names.index(link.origin_node.name), self.layers_names.index(link.target_node.name))
            desc_matrix[idx] = 1
            if link.bidirectional:
                desc_matrix[(idx[1], idx[0])] = 1

        return desc_matrix