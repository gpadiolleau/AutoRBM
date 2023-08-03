import itertools
import inspect
from functools import partial

import numpy as np

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrnd
import jax

def _get_factors_indexes(rbm_instance):
        return [i for i,layer in enumerate(rbm_instance.graph_description.nodes) if type(layer).__name__ == "LinearLayer"]

def _get_inferred_layers(rbm_instance):
    inferred_layers_mat = rbm_instance._adjacent_matrix.copy()
    inferred_layers_mat[:,rbm_instance._factors_idxs] = np.zeros_like(inferred_layers_mat[:,rbm_instance._factors_idxs])
    inferred_layers_idxs = np.where(np.sum(inferred_layers_mat,0))
    inferred_layers = list(np.asarray(rbm_instance.graph_description.nodes)[inferred_layers_idxs])
    return inferred_layers

def _analyze_graph(rbm_instance):
    rbm_instance._adjacent_matrix = rbm_instance.graph_description.get_description_as_matrix()
    rbm_instance._factors_idxs = _get_factors_indexes(rbm_instance)

    rbm_instance._no_factor_adjacent_matrix = rbm_instance._adjacent_matrix.copy()
    rbm_instance._no_factor_adjacent_matrix[rbm_instance._factors_idxs] = np.zeros_like(rbm_instance._no_factor_adjacent_matrix[rbm_instance._factors_idxs])
    rbm_instance._no_factor_adjacent_matrix[:,rbm_instance._factors_idxs] = np.zeros_like(rbm_instance._no_factor_adjacent_matrix[rbm_instance._factors_idxs]).T

    rbm_instance._not_factor_idxs = [i for j, i in enumerate([*range(len(rbm_instance.graph_description.nodes))]) if j not in rbm_instance._factors_idxs]
    rbm_instance._only_factors_adjacent_matrix = rbm_instance._adjacent_matrix.copy()
    rbm_instance._only_factors_adjacent_matrix[:,rbm_instance._not_factor_idxs] = np.zeros_like(rbm_instance._only_factors_adjacent_matrix[:,rbm_instance._not_factor_idxs])

    rbm_instance._adjacent_matrix_second_order = np.matmul(rbm_instance._only_factors_adjacent_matrix, rbm_instance._adjacent_matrix)

    rbm_instance._inferred_layers = _get_inferred_layers(rbm_instance)

    rbm_instance._neural_layers = [layer for layer in rbm_instance.graph_description.nodes if type(layer).__name__ == "NeuralLayer"]


def _create_weights(rbm_instance):
    rng_key = jrnd.PRNGKey(0)
    initializer = jnn.initializers.glorot_uniform()

    def create_weights(link):
        weights_shape = (link.origin_node.units_number, link.target_node.units_number)
        if link.initial_weights is None:
            w = initializer(rng_key, weights_shape, dtype=rbm_instance.dtype)
        elif jnp.isscalar(link.initial_weights):
            w = link.initial_weights * jrnd.normal(rng_key, weights_shape, dtype=rbm_instance.dtype)
        else:
            w = jnp.copy(link.initial_weights)
        name = f"W_{link.origin_node.name}{link.target_node.name}"
        #print(f"Create Weights {name} with shape : {w.shape}")
        return w, name

    for l in rbm_instance.graph_description.links:
        w, name = create_weights(l)
        setattr(rbm_instance, name, w)
        tmp_dict = getattr(rbm_instance, "weights_dict")
        tmp_dict[name] = w
        setattr(rbm_instance, "weights_dict",tmp_dict)

def _create_biases(rbm_instance):
    
    def create_biases(layer):
        bias_shape = (layer.units_number,)
        if layer.initial_biases is None:
            b = jnp.zeros(bias_shape, dtype=rbm_instance.dtype)
        elif jnp.isscalar(layer.initial_biases):
            b = layer.initial_biases * jnp.ones(bias_shape, dtype=rbm_instance.dtype)
        else:
            b = jnp.copy(layer.initial_biases)
        name = f"B_{layer.name}"
        #print(f"Create Biases {name} with shape {b.shape}")
        return b, name
    
    for l in rbm_instance._inferred_layers:
        b, name = create_biases(l)
        setattr(rbm_instance, name, b)
        tmp_dict = getattr(rbm_instance, "biases_dict")
        tmp_dict[name] = b
        setattr(rbm_instance, "biases_dict",tmp_dict)
        
def _create_sigmas(rbm_instance):
    
    def create_sigmas(layer):
        sigma_shape = (layer.units_number,)
        if layer.initial_sigmas is None:    
            s = jnp.ones(sigma_shape, dtype=rbm_instance.dtype)
        elif jnp.isscalar(layer.initial_sigmas):
            s = layer.initial_sigmas * jnp.ones(sigma_shape, dtype=rbm_instance.dtype) 
        else:
            s = jnp.copy(layer.initial_sigmas)
        name = f"S_{layer.name}"
        #print(f"Create Sigmas {name} with shape {s.shape}")
        return s, name
    
    for l in rbm_instance.graph_description.nodes:
        if type(l).__name__ == "NeuralLayer" and l.type == "Gaussian":
            s, name = create_sigmas(l)
            setattr(rbm_instance, name, s)
            tmp_dict = getattr(rbm_instance, "sigmas_dict")
            tmp_dict[name] = s
            setattr(rbm_instance, "sigmas_dict",tmp_dict)

def _create_instance_parameters(rbm_instance):
    setattr(rbm_instance, "weights_dict", dict())
    _create_weights(rbm_instance)
    setattr(rbm_instance, "biases_dict", dict())
    _create_biases(rbm_instance)
    setattr(rbm_instance, "sigmas_dict", dict())
    _create_sigmas(rbm_instance)


def _create_factors_fns(rbm_instance):
    needed_factors = rbm_instance._adjacent_matrix.copy()
    needed_factors = needed_factors[:,rbm_instance._factors_idxs]
    needed_factors = np.where(needed_factors)

    _to_factor = np.asarray(rbm_instance.graph_description.nodes)[np.asarray(rbm_instance._factors_idxs)[needed_factors[1]]]
    _for_layer = np.asarray(rbm_instance.graph_description.nodes)[needed_factors[0]]

    def create_factor_fn(input_layer, factor_layer):
        fn_name = f"factorize_{input_layer.name}_on_{factor_layer.name}" 
        arg_mapping = {f"{input_layer.name}": "inputs"}

        newline = "\n"
        indent = "\t"
        docstring_template = f"""Computes message passing for layer '{input_layer.name}' to factor '{factor_layer.name}'{newline}"""
        docstring_template += f"""{newline}Args:{newline+indent}{input_layer.name} : Array of shape ({input_layer.units_number},) representing values for inputs vector of layer '{input_layer.name}'{newline}"""
        docstring_template += f"""{newline}Returns:{newline+indent}array : Message values vector to factor '{factor_layer.name}' from '{input_layer.name}' with shape ({factor_layer.units_number},){newline}"""
        docstring_template += f"""{newline}Raises:{newline+indent}ValueError: Input vector has wrong shape"""

        def factor_fn(*args, **kwargs):
            if args:
                inputs = args[0]
            else:
                kwargs = {arg_mapping.get(key, key): value for key, value in kwargs.items()}
                inputs = kwargs["inputs"]  # Extract the renamed argument value

            try:
                assert inputs.shape == (input_layer.units_number,)
            except:
                raise ValueError(f"Expected shape for {input_layer.name} is ({input_layer.units_number},), but received {inputs.shape}")

            w_name = f"W_{input_layer.name}{factor_layer.name}"
            w = getattr(rbm_instance, w_name)
            return jnp.matmul(inputs, w)
        
        # Créer la signature de la fonction
        factor_fn_args = [inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD) for arg in arg_mapping.keys()]
        factor_fn_signature = inspect.Signature(parameters=factor_fn_args)
        factor_fn.__signature__ = factor_fn_signature

        
        #print(f"Create factor_fn {fn_name}")
        factor_fn.__doc__ = docstring_template
        setattr(rbm_instance, fn_name, jax.jit(factor_fn))
        setattr(rbm_instance, f"batched_{fn_name}", jax.jit(jax.vmap(factor_fn))) # add batched version of the fonction
    
    for f,l in zip(_to_factor, _for_layer):
        create_factor_fn(l,f)

def _create_get_probs_fns(rbm_instance):

    def create_get_probs_fn(layer):

        fn_name = f"get_probabilities_{layer.name}"
        #print(f"Create {fn_name}() function for layer {layer.name}")

        layer_idx = rbm_instance.graph_description.nodes.index(layer)

        _sanitized_amat_second_order = rbm_instance._adjacent_matrix_second_order.copy()   
        _sanitized_amat_second_order[layer_idx] = np.zeros_like(_sanitized_amat_second_order[layer_idx])

        first_order_inputs  = list(np.asarray(rbm_instance.graph_description.nodes)[np.where(rbm_instance._no_factor_adjacent_matrix[:,layer_idx]>0)])
        second_order_inputs = list(np.asarray(rbm_instance.graph_description.nodes)[np.where(_sanitized_amat_second_order[:,layer_idx]>0)])
        all_inputs = first_order_inputs+second_order_inputs
        inputs_names = [*set([l.name for l in all_inputs])]

        all_inputs = [layer for layer in rbm_instance.graph_description.nodes if layer.name in inputs_names]
        layer.inputs = all_inputs

        only_factors_adjacent_matrix = rbm_instance._only_factors_adjacent_matrix.copy()

        which_factors_idxs = np.where(only_factors_adjacent_matrix[layer_idx])

        #print(f"Inputs are : {inputs_names}")
        #print(f"First order inputs : {[l.name for l in first_order_inputs]}")
        #print(f"Second order inputs : {[l.name for l in second_order_inputs]}")

        arg_mapping = {f"{name}": f"input_{i+1}" for i,name in enumerate(inputs_names)}
        #print(f"arg mapping is : {arg_mapping}")

        newline = "\n"
        indent = "\t"
        docstring_template = f"""Computes conditional probabilities for layer '{layer.name}' given {', '.join(inputs_names)}{newline}"""
        docstring_template += f"""{newline}Args:{newline+indent}{(newline+indent).join([f"{input_layer.name} : Array of shape ({input_layer.units_number},) representing values for input vector of layer '{input_layer.name}'" for input_layer in all_inputs])}{newline}"""
        docstring_template += f"""{newline}Returns:{newline+indent}array : Condtional probabilities {layer.name} given {', '.join(inputs_names)} with shape ({layer.units_number},)"""
        docstring_template += f"""{newline}Raises:{newline+indent}ValueError: Any of first order input vectors has wrong shape"""

        def probs_fn(*args, **kwargs):
            # check inputs
            fn_args = {k:None for k in arg_mapping.keys()}
            if len(args) > 0:
                for i,arg in enumerate(args):
                    fn_args[[*arg_mapping.keys()][i]] = arg

            if len([*kwargs]) > 0:
                for k,v in kwargs.items():
                    fn_args[k] = v

            raise_error = False
            for k,v in fn_args.items():
                try:
                    if v.all() == None:
                        raise_error = True
                        break
                except:
                    if v == None:
                        raise_error =True
                        break
            if raise_error:
                raise TypeError(f"{fn_name}() missing required positional argument '{k}'")
            
            ## Build the function
            # compute first order terms
            first_order_term = jnp.zeros((layer.units_number,))
            for input_layer in first_order_inputs:
                term_input = fn_args[input_layer.name]
                try:
                    assert term_input.shape == (input_layer.units_number,)
                except:
                    raise ValueError(f"""Expected shape for input '{input_layer.name}' is ({input_layer.units_number},), but received {term_input.shape}""")
                try:
                    w = getattr(rbm_instance, f"W_{input_layer.name}{layer.name}")
                except:
                    w = getattr(rbm_instance, f"W_{layer.name}{input_layer.name}")
                    w = w.T
                term = jnp.matmul(term_input, w)
                first_order_term += term
            # Compute second order terms   
            second_order_term = jnp.zeros((layer.units_number,))
            if second_order_inputs:
                for j,i in enumerate(which_factors_idxs[0]):
                    factor_j_inputs_idxs = only_factors_adjacent_matrix.copy()[:,i]
                    factor_j_inputs_idxs[layer_idx] = 0.
                    factor_j_inputs_idxs = np.where(factor_j_inputs_idxs)[0]

                    factor_layer = rbm_instance.graph_description.nodes[i]
                    #print(f"Factor {[l.name for l in list(np.array(rbm_instance.graph_description.nodes)[which_factors_idxs])][j]} "
                    #    f"has inputs {[l.name for l in list(np.asarray(rbm_instance.graph_description.nodes)[factor_j_inputs_idxs])]} for {layer.name}")
                    factor_j_term = jnp.ones((factor_layer.units_number,))
                    for input_layer in list(np.asarray(rbm_instance.graph_description.nodes)[factor_j_inputs_idxs]):

                        factor_fn_name = f"factorize_{input_layer.name}_on_{factor_layer.name}"
                        #print(f"Calling {factor_fn_name}")
                        factor_fn = getattr(rbm_instance, factor_fn_name)
                        factor_j_term *= factor_fn(fn_args[input_layer.name])
                        
                    w = getattr(rbm_instance, f"W_{layer.name}{factor_layer.name}")
                    factor_j_term = jnp.matmul(factor_j_term, w.T)
                    second_order_term += factor_j_term

            # Compute bias term
            try:
                bias_term = getattr(rbm_instance, f"B_{layer.name}")
            except:
                bias_term = jnp.zeros((layer.units_number,))

            
            preactivation = first_order_term + second_order_term + bias_term
            if layer.type == "Binary": activation_fn = rbm_instance.sigmoid_activation
            elif layer.type == "Gaussian": activation_fn = rbm_instance.linear_activation
            elif layer.type == "ReLU": activation_fn = rbm_instance.softplus_activation

            return activation_fn(preactivation) 


        # Créer la signature de la fonction
        probs_fn_args = [inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD) for arg in arg_mapping.keys()]
        probs_fn_signature = inspect.Signature(parameters=probs_fn_args)
        probs_fn.__signature__ = probs_fn_signature     
            
        probs_fn.__doc__ = docstring_template
        setattr(rbm_instance, fn_name, jax.jit(probs_fn))
        setattr(rbm_instance, f"batched_{fn_name}", jax.jit(jax.vmap(probs_fn))) # add batched version of the fonction

    for l in rbm_instance._inferred_layers:
        create_get_probs_fn(l)

def _create_samples_fns(rbm_instance):
    
    def create_samples_fn(layer):
        fn_name = f"sample_{layer.name}"

        arg_mapping = {f"probs_{layer.name}":"inputs"}

        newline = "\n"
        indent = "\t"
        docstring_template = f"""Samples probabilities probs_{layer.name}{newline}"""
        docstring_template += f"""{newline}Args:{newline+indent}probs_{layer.name}: Array of shape ({layer.units_number},) representing probabilities values of layer '{layer.name}' {newline}"""
        docstring_template += f"""{newline}Returns:{newline+indent}array : Samples for {layer.name} with shape ({layer.units_number},)"""

        def sample_fn(*args, **kwargs):
            if args:
                inputs = args[0]
            else:
                kwargs = {arg_mapping.get(key, key): value for key, value in kwargs.items()}
                inputs = kwargs["inputs"]  # Extract the renamed argument value
            
            if layer.type == "Binary": 
                rng_key = jrnd.PRNGKey(0)
                boolean_samples = jrnd.bernoulli(rng_key, inputs)
                samples = jnp.array(boolean_samples, dtype=rbm_instance.dtype)
            elif layer.type == "Gaussian": 
                rng_key = jrnd.PRNGKey(0)
                scale = getattr(rbm_instance, f"S_{layer.name}")
                scale = jnp.square(scale)
                samples = inputs + jrnd.normal(rng_key, shape=inputs.shape, dtype=rbm_instance.dtype) * scale
            elif layer.type == "ReLU": 
                rng_key = jrnd.PRNGKey(0)
                samples = (inputs + jrnd.normal(rng_key, shape=inputs.shape, dtype=rbm_instance.dtype) * scale) * rbm_instance.sigmoid_activation(inputs)
                samples = jnp.clip(samples, 0.0, rbm_instance.max_act)

            return samples
        
        # Créer la signature de la fonction
        sample_fn_args = [inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD) for arg in arg_mapping.keys()]
        sample_fn_signature = inspect.Signature(parameters=sample_fn_args)
        sample_fn.__signature__ = sample_fn_signature     
            
        sample_fn.__doc__ = docstring_template       
        setattr(rbm_instance, fn_name, jax.jit(sample_fn))
        setattr(rbm_instance, f"batched_{fn_name}", jax.jit(jax.vmap(sample_fn))) # add batched version of the fonction
    
    def create_get_samples_fn(layer):
        fn_name = f"get_samples_{layer.name}"
        layer_idx = rbm_instance.graph_description.nodes.index(layer)

        sanitized_amat_second_order = rbm_instance._adjacent_matrix_second_order.copy()
        sanitized_amat_second_order[layer_idx] = np.zeros_like(sanitized_amat_second_order[layer_idx])

        first_order_inputs  = list(np.asarray(rbm_instance.graph_description.nodes)[np.where(rbm_instance._no_factor_adjacent_matrix[:,layer_idx]>0)])
        second_order_inputs = list(np.asarray(rbm_instance.graph_description.nodes)[np.where(sanitized_amat_second_order[:,layer_idx]>0)])
        all_inputs = first_order_inputs+second_order_inputs
        inputs_names = [*set([l.name for l in all_inputs])]

        all_inputs = [layer for layer in rbm_instance.graph_description.nodes if layer.name in inputs_names]

        arg_mapping = {f"{name}": f"input_{i+1}" for i,name in enumerate(inputs_names)}

        newline = "\n"
        indent = "\t"
        docstring_template = f"""Computes samples for layer '{layer.name}' given {', '.join(inputs_names)}{newline}"""
        docstring_template += f"""{newline}Args:{newline+indent}{(newline+indent).join([f"{input_layer.name} : Array of shape ({input_layer.units_number},) representing values for input vector of layer '{input_layer.name}'" for input_layer in all_inputs])}{newline}"""
        docstring_template += f"""{newline}Returns:{newline+indent}array : Samples for {layer.name} given {', '.join(inputs_names)} with shape ({layer.units_number},)"""

        def get_samples_fn(*args, **kwargs):
            # check inputs
            fn_args = {k:None for k in arg_mapping.keys()}
            if len(args) > 0:
                for i,arg in enumerate(args):
                    fn_args[[*arg_mapping.keys()][i]] = arg

            if len([*kwargs]) > 0:
                for k,v in kwargs.items():
                    fn_args[k] = v

            raise_error = False
            for k,v in fn_args.items():
                try:
                    if v.all() == None:
                        raise_error = True
                        break
                except:
                    if v == None:
                        raise_error =True
                        break
            if raise_error:
                raise TypeError(f"{fn_name}() missing required positional argument '{k}'")
            
            get_probs_fn_name = f"get_probabilities_{layer.name}"
            sample_fn_name = f"sample_{layer.name}"
            probs_fn = getattr(rbm_instance, get_probs_fn_name)
            sample_fn = getattr(rbm_instance, sample_fn_name)

            probs = probs_fn(*fn_args.values())
            samples = sample_fn(probs)
            return samples

        # Créer la signature de la fonction
        get_samples_fn_args = [inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD) for arg in arg_mapping.keys()]
        get_samples_fn_signature = inspect.Signature(parameters=get_samples_fn_args)
        get_samples_fn.__signature__ = get_samples_fn_signature     
            
        get_samples_fn.__doc__ = docstring_template
        setattr(rbm_instance, fn_name, jax.jit(get_samples_fn))
        setattr(rbm_instance, f"batched_{fn_name}", jax.jit(jax.vmap(get_samples_fn))) # add batched version of the fonction

    for l in rbm_instance._inferred_layers:
        create_samples_fn(l)
        create_get_samples_fn(l)

def _create_get_energy_fn(rbm_instance):
    fn_name = "get_energy"
    #print(f"Create {fn_name} function")

    inputs = [l for l in rbm_instance._neural_layers]
    inputs_names = [l.name for l in inputs]
    parameters_dict_names = ["weights_dict", "biases_dict", "sigmas_dict"]
    args_names = inputs_names + parameters_dict_names

    newline = "\n"
    indent = "\t"
    docstring_template = f"""Computes energy value given states of layers{newline}"""
    docstring_template += f"""{newline}Args:{newline+indent}{(newline+indent).join([f"{input_layer.name} : Array of shape ({input_layer.units_number},) representing values for input vector of layer '{input_layer.name}'" for input_layer in inputs])}{newline}"""
    docstring_template += f"""{newline}Returns:{newline+indent}float : Energy value given {', '.join(inputs_names)}"""


    first_order_interactions_matrix = rbm_instance._no_factor_adjacent_matrix * np.triu(np.ones_like(rbm_instance._no_factor_adjacent_matrix))
    first_order_interactions_matrix += (rbm_instance._no_factor_adjacent_matrix * np.tril(np.ones_like(rbm_instance._no_factor_adjacent_matrix))).T

    second_order_interactions_matrix = rbm_instance._only_factors_adjacent_matrix * np.triu(np.ones_like(rbm_instance._only_factors_adjacent_matrix))
    second_order_interactions_matrix += (rbm_instance._only_factors_adjacent_matrix * np.tril(np.ones_like(rbm_instance._only_factors_adjacent_matrix)))

    def _get_energy_fn(*args, **kwargs):
        # check inputs
        fn_args = {k:None for k in args_names}
        if len(args) > 0:
            for i,arg in enumerate(args):
                fn_args[args_names[i]] = arg

        if len([*kwargs]) > 0:
            for k,v in kwargs.items():
                fn_args[k] = v

        raise_error = False
        for k,v in fn_args.items():
            try:
                if v.all() == None:
                    raise_error = True
                    break
            except:
                if v == None:
                    raise_error =True
                    break
        if raise_error:
            raise TypeError(f"{fn_name}() missing required positional argument '{k}'")

        first_order_interactions_energy_term = 0.
        for fo_idx in np.array(np.where(first_order_interactions_matrix)).T:
            first_layer = rbm_instance.graph_description.nodes[fo_idx[0]]
            second_layer = rbm_instance.graph_description.nodes[fo_idx[1]]
            #print(f"First Order interaction energy term between {first_layer.name} and {second_layer.name}")
            try:
                w = fn_args['weights_dict'][f"W_{first_layer.name}{second_layer.name}"]
                #w = getattr(rbm_instance, f"W_{first_layer.name}{second_layer.name}")
            except:
                w = fn_args['weights_dict'][f"W_{second_layer.name}{first_layer.name}"]
                #w = getattr(rbm_instance, f"W_{second_layer.name}{first_layer.name}")
                w = w.T
            first_arg = fn_args[first_layer.name]
            if first_layer.type == "Gaussian":
                s = fn_args["sigmas_dict"][f"S_{first_layer.name}"]
                #s = getattr(rbm_instance, f"S_{first_layer.name}")
                first_arg = jnp.divide(first_arg,jnp.square(s))
            second_arg = fn_args[second_layer.name]
            if second_layer.type == "Gaussian":
                s = fn_args["sigmas_dict"][f"S_{second_layer.name}"]
                #s = getattr(rbm_instance, f"S_{second_layer.name}")
                second_arg = jnp.divide(second_arg,jnp.square(s))
            #print(f"{first_layer.name} of shape {fn_args[first_layer.name].shape}, w of shape {w.shape}, {second_layer.name} of shape {fn_args[second_layer.name].shape}")
            eterm = -jnp.matmul(jnp.matmul(first_arg.T, w), second_arg)
            first_order_interactions_energy_term += eterm
            #print(first_order_interactions_energy_term)

        second_order_interactions_energy_term = 0.
        for factor_idx in rbm_instance._factors_idxs:
            factor_layer = rbm_instance.graph_description.nodes[factor_idx]
            factor_energy_term = jnp.ones((factor_layer.units_number,))
            for so_idx in np.argwhere(second_order_interactions_matrix[:,factor_idx]):
                #print(f"Second Order interaction energy term between {rbm_instance.graph_description.nodes[factor_idx].name} and {rbm_instance.graph_description.nodes[so_idx[0]].name}")
                factor_input_layer = rbm_instance.graph_description.nodes[so_idx[0]]
                factor_input_arg = fn_args[factor_input_layer.name]
                if factor_input_layer.type == "Gaussian":
                    s = fn_args["sigmas_dict"][f"S_{factor_input_layer.name}"]
                    #s = getattr(rbm_instance, f"S_{factor_input_layer.name}")
                    factor_input_arg = jnp.divide(factor_input_arg, jnp.square(s))
                w = fn_args["weights_dict"][f"W_{factor_input_layer.name}{factor_layer.name}"]
                #w = getattr(rbm_instance, f"W_{factor_input_layer.name}{factor_layer.name}")
                eterm = jnp.matmul(factor_input_arg, w)
                factor_energy_term *= eterm
            second_order_interactions_energy_term += -jnp.sum(factor_energy_term)
            #print(second_order_interactions_energy_term)
        
        bias_energy_term = 0.
        for layer in rbm_instance._inferred_layers:
            bias_arg = fn_args[layer.name]
            b = fn_args["biases_dict"][f"B_{layer.name}"]
            #b = getattr(rbm_instance, f"B_{layer.name}")
            if layer.type == "Gaussian":
                s = fn_args["sigmas_dict"][f"S_{layer.name}"]
                #s = getattr(rbm_instance, f"S_{layer.name}")
                eterm = 0.5*jnp.sum(jnp.square(jnp.divide(bias_arg-b,s)))
            else:
                eterm = -jnp.dot(bias_arg, b.T)
            bias_energy_term += eterm
            #print(bias_energy_term)

        energy  = first_order_interactions_energy_term + second_order_interactions_energy_term + bias_energy_term
        return energy

    # Créer la signature de la fonction
    get_energy_fn_args = [inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD) for arg in inputs_names]
    _get_energy_fn_args = get_energy_fn_args + [inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD) for arg in ["weights_dict","biases_dict","sigmas_dict"]]
    get_energy_fn_signature = inspect.Signature(parameters=get_energy_fn_args)
    _get_energy_fn_signature = inspect.Signature(parameters=_get_energy_fn_args)
    _get_energy_fn.__signature__ = _get_energy_fn_signature
       
         
    _get_energy_fn.__doc__ = docstring_template
    setattr(rbm_instance, f"_{fn_name}", _get_energy_fn)

    partial_get_energy_fn = partial(_get_energy_fn, 
                                    weights_dict=getattr(rbm_instance,"weights_dict"),
                                    biases_dict=getattr(rbm_instance, "biases_dict"), 
                                    sigmas_dict=getattr(rbm_instance,"sigmas_dict"))
    partial_get_energy_fn.__signature__ = get_energy_fn_signature
    partial_get_energy_fn.__doc__ = docstring_template
    setattr(rbm_instance, fn_name, jax.jit(partial_get_energy_fn))

    setattr(rbm_instance, f"batched_{fn_name}", jax.jit(jax.vmap(partial_get_energy_fn, in_axes=[0]*len(inputs)))) # add batched version of the fonction

    def _batched_get_energy_fn(*args, **kwargs):
        # check inputs
        fn_args = {k:None for k in args_names}
        if len(args) > 0:
            for i,arg in enumerate(args):
                fn_args[args_names[i]] = arg

        if len([*kwargs]) > 0:
            for k,v in kwargs.items():
                fn_args[k] = v

        raise_error = False
        for k,v in fn_args.items():
            try:
                if v.all() == None:
                    raise_error = True
                    break
            except:
                if v == None:
                    raise_error =True
                    break
        if raise_error:
            raise TypeError(f"{fn_name}() missing required positional argument '{k}'")
        partial_fn = partial(_get_energy_fn,
                             weights_dict=fn_args["weights_dict"], biases_dict=fn_args["biases_dict"], sigmas_dict=fn_args["sigmas_dict"])
        
        return jax.vmap(partial_fn)(*[fn_args[a] for a in inputs_names])
    
    def _meanbatched_get_energy_fn(*args, **kwargs):
        # check inputs
        fn_args = {k:None for k in args_names}
        if len(args) > 0:
            for i,arg in enumerate(args):
                fn_args[args_names[i]] = arg

        if len([*kwargs]) > 0:
            for k,v in kwargs.items():
                fn_args[k] = v

        raise_error = False
        for k,v in fn_args.items():
            try:
                if v.all() == None:
                    raise_error = True
                    break
            except:
                if v == None:
                    raise_error =True
                    break
        if raise_error:
            raise TypeError(f"{fn_name}() missing required positional argument '{k}'")
        return jnp.mean(_batched_get_energy_fn(*[fn_args[a] for a in args_names]))

    setattr(rbm_instance, f"_batched_{fn_name}", _batched_get_energy_fn)
    setattr(rbm_instance, f"_get_energy_grad", jax.grad(_get_energy_fn, [*range(len(inputs_names), len(inputs_names)+len(parameters_dict_names))]))
    setattr(rbm_instance, f"_batched_get_energy_grad", jax.grad(_meanbatched_get_energy_fn, [*range(len(inputs_names), len(inputs_names)+len(parameters_dict_names))]))

    def get_energy_grad_fn(*args, **kwargs):
        # check inputs
        fn_args = {k:None for k in inputs_names}
        if len(args) > 0:
            for i,arg in enumerate(args):
                fn_args[inputs_names[i]] = arg

        if len([*kwargs]) > 0:
            for k,v in kwargs.items():
                fn_args[k] = v

        raise_error = False
        for k,v in fn_args.items():
            try:
                if v.all() == None:
                    raise_error = True
                    break
            except:
                if v == None:
                    raise_error =True
                    break
        if raise_error:
            raise TypeError(f"{fn_name}() missing required positional argument '{k}'")
        return getattr(rbm_instance, "_get_energy_grad")(*[fn_args[a] for a in inputs_names], 
                                                         getattr(rbm_instance, parameters_dict_names[0]),
                                                         getattr(rbm_instance, parameters_dict_names[1]),
                                                         getattr(rbm_instance, parameters_dict_names[2]))
    
    setattr(rbm_instance, f"get_energy_grad", jax.jit(get_energy_grad_fn))

    def batched_get_energy_grad_fn(*args, **kwargs):
        # check inputs
        fn_args = {k:None for k in inputs_names}
        if len(args) > 0:
            for i,arg in enumerate(args):
                fn_args[inputs_names[i]] = arg

        if len([*kwargs]) > 0:
            for k,v in kwargs.items():
                fn_args[k] = v

        raise_error = False
        for k,v in fn_args.items():
            try:
                if v.all() == None:
                    raise_error = True
                    break
            except:
                if v == None:
                    raise_error =True
                    break
        if raise_error:
            raise TypeError(f"{fn_name}() missing required positional argument '{k}'")
        return getattr(rbm_instance, "_batched_get_energy_grad")(*[fn_args[a] for a in inputs_names], 
                                                         getattr(rbm_instance, parameters_dict_names[0]),
                                                         getattr(rbm_instance, parameters_dict_names[1]),
                                                         getattr(rbm_instance, parameters_dict_names[2]))
    
    setattr(rbm_instance, f"batched_get_energy_grad", jax.jit(batched_get_energy_grad_fn))

def _create_propagation_fns(rbm_instance):

    
    def stabilized_propagation(inputs, sorted_layers, priorities_dict, K_steps=5, K_meta_steps=5, eps_stab=1e-10, verbose=False):

        def stabilization_step(prio, layers, upper_scope_locals):
            # for upper prio : compute propagation through the graph
        
            if verbose: print(f"P={prio} : Compute propagation")
            if layers:
                # if only one layer on stage/prio 
                if len(layers) == 1: # assuming this layer is inferred
                    l = layers[0]
                    if verbose: print(f"> Compute {l.name} samples and probabilities")
                    # get fns
                    probs_fn = getattr(rbm_instance, f"get_probabilities_{l.name}")
                    probs_fn_signature = inspect.signature(probs_fn)
                    probs_args_names = list(probs_fn_signature.parameters.keys())
                    
                    upper_scope_locals[f"{l.name}_probs"] = probs_fn(*[upper_scope_locals[f"{k}_samples"] for k in probs_args_names])

                    sample_fn = getattr(rbm_instance, f"sample_{l.name}")

                    upper_scope_locals[f"{l.name}_samples"] = sample_fn(upper_scope_locals[f"{l.name}_probs"])
                    
                # if more than one layer on stage/prio
                if len(layers) > 1:
                    if verbose: print(f"More than one layer on stage {prio}.")
                    # check for co-dependant layers
                    layers_indexes = [rbm_instance.graph_description.nodes.index(l) for l in layers]
                    combinations_layers_indexes = list(itertools.combinations(layers_indexes, 2))
                    for c in combinations_layers_indexes:
                        codependency = (rbm_instance._adjacent_matrix + rbm_instance._adjacent_matrix_second_order)[c] == \
                                    (rbm_instance._adjacent_matrix + rbm_instance._adjacent_matrix_second_order).T[c] != 0
                        # if codependency betwen two or more layers on stage -> need stabilization through Gibbs sampling
                        if codependency :
                            if verbose: print("Codependency detected ! Stabilization required.")
                            k_init = 0
                            eps_init = jnp.inf
                            clayers = [rbm_instance.graph_description.nodes[cc] for cc in c \
                                       if rbm_instance.graph_description.nodes[cc] in rbm_instance._inferred_layers] # codependent layers
                            samples_fns = [getattr(rbm_instance, f"get_samples_{l.name}") for l in clayers]
                            samples_fns_args_names = [list(inspect.signature(samples_fn).parameters.keys()) for samples_fn in samples_fns]
                            samples_fns_args = [[upper_scope_locals[f"{k}_samples"] for k in samples_args_names] for samples_args_names in samples_fns_args_names]
                            locals_utils = [upper_scope_locals[f"{l.name}_samples"] for l in clayers]

                            def cond_fn(state):
                                k, eps, _ = state
                                return jnp.logical_and(k < K_steps, eps > eps_stab)

                            def while_loop_body_fn(k_eps_locals_utils):
                                k, eps, locals_utils = k_eps_locals_utils
                                _eps = jnp.ones(len(clayers))
                                for i in range(len(clayers)):
                                    l = clayers[i]
                                    if verbose:
                                        print(f"> Compute {l.name} samples")
                                    prev_val = locals_utils[i].copy()
                                    new_val = samples_fns[i](*samples_fns_args[i])
                                    locals_utils[i] = new_val
                                    _eps = _eps.at[i].set(jnp.sum(jnp.abs(prev_val - new_val)))

                                k += 1
                                eps = jnp.sum(_eps)

                                return (k, eps, locals_utils)
                            
                            k, eps, locals_utils = jax.lax.while_loop(cond_fn, while_loop_body_fn, (k_init, eps_init, locals_utils))
                            if verbose: print(f"Stabilization reaches eps:{eps} in {k} steps")
                            for i,l in enumerate(clayers):
                                upper_scope_locals[f"{l.name}_samples"] = locals_utils[i]

                    # Stabilization is over, compute states for all layers on stage
                    if verbose: print("Final stabilization step")
                    for l in layers:
                        if l in rbm_instance._inferred_layers:
                            if verbose: print(f"> Compute {l.name} samples and probabilities")
                            # get fns
                            probs_fn = getattr(rbm_instance, f"get_probabilities_{l.name}")
                            probs_fn_signature = inspect.signature(probs_fn)
                            probs_args_names = list(probs_fn_signature.parameters.keys())
                            _locals = upper_scope_locals
                            upper_scope_locals[f"{l.name}_probs"] = probs_fn(*[_locals[f"{k}_samples"] for k in probs_args_names])
                            sample_fn = getattr(rbm_instance, f"sample_{l.name}")
                            upper_scope_locals[f"{l.name}_samples"] = sample_fn(upper_scope_locals[f"{l.name}_probs"])

            return upper_scope_locals  

        # initialize probs and samples local variables for all layers
        for l in sorted_layers:
            if l.name in inputs.keys():
                if verbose: print(f"> Layer {l.name} takes values of placeholder {l.name}")
                locals()[f"{l.name}_probs"] = inputs[l.name]
                locals()[f"{l.name}_samples"] = inputs[l.name]
            else:
                locals()[f"{l.name}_probs"] = jnp.zeros((l.units_number,))
                locals()[f"{l.name}_samples"] = jnp.zeros_like(locals()[f"{l.name}_probs"])

        for prio, layers in priorities_dict.items():
            if prio > sorted_layers[0].priority:
                returned_vars = stabilization_step(prio, layers, locals())
                # assign returned vars to locals
                for l in layers:
                    locals()[f"{l.name}_probs"] = returned_vars[f"{l.name}_probs"] 
                    locals()[f"{l.name}_samples"] = returned_vars[f"{l.name}_samples"]
        
        # hidden layers stabilization   
        if verbose: print("Meta Stabilization")
        kk=0
        Eps_dict = {prio:jnp.inf for prio in priorities_dict.keys()}
        Eps_dict[0] = 0.
        Eps = jnp.sum(jnp.array([v for v in Eps_dict.values()]))

        def cond_fn(state):
            k, eps, _ = state
            return jnp.logical_and(k < K_meta_steps, eps > eps_stab)
        
        def while_loop_body_fn(k_eps_locals_utils):
            k, eps, locals_utils = k_eps_locals_utils
            for i, (prio, layers) in enumerate(priorities_dict.items()):
                if prio > 0:
                    prev_vals = list()
                    for l in layers:
                        prev_vals.append(locals_utils[f"{l.name}_samples"])
                    returned_vars = stabilization_step(prio, layers, locals_utils)
                    # assign returned vars to locals
                    for l in layers:
                        locals_utils[f"{l.name}_probs"] = returned_vars[f"{l.name}_probs"] 
                        locals_utils[f"{l.name}_samples"] = returned_vars[f"{l.name}_samples"]
                    new_vals = list()
                    for l in layers:
                        new_vals.append(locals_utils[f"{l.name}_samples"] )
                    Eps_dict[prio] = jnp.sum(jnp.array([jnp.sum(jnp.abs(pv-nv)) for pv,nv in zip(prev_vals, new_vals)]))
                    if verbose: print(f"Eps value for stage {prio}: {Eps_dict[prio]}")
                    
            eps = jnp.sum(jnp.array([v for v in Eps_dict.values()]))
            k += 1
            eps = jnp.sum(eps)

            return (k, eps, locals_utils)

        init_locals_utils = {k:jnp.array(v) for k,v in locals().items() if ("_samples" in k or "_probs" in k)}
        kk, Eps, locals_utils = jax.lax.while_loop(cond_fn, while_loop_body_fn, (kk, Eps, init_locals_utils))
        if verbose: print(f"Meta-Stabilization reaches eps:{Eps} in {kk} steps")
        probs = list()
        samples = list()
        for l in sorted_layers:
            probs.append(locals_utils[f"{l.name}_probs"])
            samples.append(locals_utils[f"{l.name}_samples"])

        return probs, samples
    
    bottom_up_fn_name = "bottom_up_propagation"
    bottom_up_inputs = [l for l in rbm_instance._neural_layers if l.priority == min(rbm_instance._neural_layers, key=lambda l: l.priority).priority]
    bottom_up_inputs_names_mandatory = [l.name for l in bottom_up_inputs]
    bottom_up_inputs_names_optional = ["K_steps", "K_meta_steps", "eps_stab", "verbose"]
    bottom_up_inputs_names = bottom_up_inputs_names_mandatory+bottom_up_inputs_names_optional

    newline = "\n"
    indent = "\t"
    docstring = f"""Computes a bottom-up propagation through the graph."""
    docstring += f"""{newline}Args:{newline+indent}{(newline+indent).join([f"{arg_name} : Array of same shape as {arg_name} layer, representing values for input vector of layer '{arg_name}'" for arg_name in bottom_up_inputs_names_mandatory])}"""
    docstring += f"""{newline+indent}K_steps (int) : Number of stabilization steps for inner graph stage stabilization if there are co-dependent layers. Default value to 5."""
    docstring += f"""{newline+indent}K_meta_steps (int) : Number of stabilization steps for hidden layers stack. Default value to 5."""
    docstring += f"""{newline+indent}eps_stab (float) : Minimum difference value between pre- and post-stabilization steps layers states used for early stopping stabilization. Default value to 1e-10."""
    docstring += f"""{newline+indent}verbose (bool) : Set verbosity of the function mainly for debug purposes and see the propagation process. Default value to False.{newline}"""
    docstring += f"""{newline}Returns:{newline+indent} dict[dict] : A dictionnary of layers states containning probabilities and sampled states. 
                      The dictionnary takes structure : dict('layer_name': dict('probabilities':probs, 'samples':sampled_states))"""


    def bottom_up_propagation_fn(*args, K_steps=5, K_meta_steps=5, eps_stab=1e-10, verbose=False, **kwargs):        
        # check inputs
        fn_args = {k:None for k in bottom_up_inputs_names}
        fn_args['K_steps'] = K_steps
        fn_args['K_meta_steps'] = K_meta_steps
        fn_args['eps_stab'] = eps_stab
        fn_args['verbose'] = verbose
        if len(args) > 0:
            for i,arg in enumerate(args):
                fn_args[bottom_up_inputs_names[i]] = arg

        if len([*kwargs]) > 0:
            for k,v in kwargs.items():
                fn_args[k] = v


        raise_error = False
        for k,v in fn_args.items():
            try:
                if v.all() == None:
                    raise_error = True
                    break
            except:
                if v == None:
                    raise_error =True
                    break
        if raise_error:
            raise TypeError(f"{bottom_up_fn_name}() missing required positional argument '{k}'")
        # bottom-up propagation
        sorted_layers = sorted(rbm_instance._neural_layers, key=lambda l: l.priority)
        lowest_piority = min(sorted_layers, key=lambda l: l.priority).priority
        priorities = list(set([l.priority-lowest_piority for l in sorted_layers]))
        priorities_dict = {k:list() for k in priorities}
        
        for l in sorted_layers:
            priorities_dict[l.priority-lowest_piority].append(l)
        inputs = {name:fn_args[name] for name in bottom_up_inputs_names}
        states = stabilized_propagation(inputs, sorted_layers, priorities_dict, K_steps, K_meta_steps, eps_stab, verbose)
        states_dict = {l.name:{"probabilities":states[0][i], "samples":states[1][i]} for i,l in enumerate(sorted_layers)}
        return states_dict
    
    # Créer la signature de la fonction
    fn_signature = inspect.signature(bottom_up_propagation_fn)
    bottom_up_propagation_fn_args = []
    for arg in bottom_up_inputs_names_mandatory:
        param = inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        bottom_up_propagation_fn_args.append(param)

    for arg in fn_signature.parameters.keys():
        if arg not in ['args', 'kwargs']:
            default_value = fn_signature.parameters[arg].default
            param = inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default_value)
            bottom_up_propagation_fn_args.append(param)
    bottom_up_propagation_fn_signature = inspect.Signature(parameters=bottom_up_propagation_fn_args)
    bottom_up_propagation_fn.__signature__ = bottom_up_propagation_fn_signature
    bottom_up_propagation_fn.__doc__ = docstring
    
    setattr(rbm_instance, bottom_up_fn_name, jax.jit(bottom_up_propagation_fn, static_argnames=bottom_up_inputs_names_optional))

    def bacthed_bottom_up_propagation_fn(*args, K_steps=5, K_meta_steps=5, eps_stab=1e-10, **kwargs):
        fn_args = {k:None for k in bottom_up_inputs_names}
        fn_args['K_steps'] = K_steps
        fn_args['K_meta_steps'] = K_meta_steps
        fn_args['eps_stab'] = eps_stab
        fn_args['verbose'] = False
        if len(args) > 0:
            for i,arg in enumerate(args):
                fn_args[bottom_up_inputs_names[i]] = arg

        if len([*kwargs]) > 0:
            for k,v in kwargs.items():
                fn_args[k] = v


        raise_error = False
        for k,v in fn_args.items():
            try:
                if v.all() == None:
                    raise_error = True
                    break
            except:
                if v == None:
                    raise_error =True
                    break
        if raise_error:
            raise TypeError(f"batched_{top_down_fn_name}() missing required positional argument '{k}'")
        
        partial_fn = partial(getattr(rbm_instance,bottom_up_fn_name), 
                             K_steps=fn_args['K_steps'], K_meta_steps=fn_args['K_meta_steps'], 
                             eps_stab=fn_args['eps_stab'], verbose=fn_args['verbose'])
        return jax.vmap(partial_fn, in_axes=[0,0])(*[fn_args[a] for a in bottom_up_inputs_names_mandatory])


    bacthed_bottom_up_propagation_fn.__signature__ = bottom_up_propagation_fn_signature
    bacthed_bottom_up_propagation_fn.__doc__ = f"""** Batched version of {bottom_up_fn_name} using partial vmap **{newline+newline}"""+docstring

    setattr(rbm_instance, f"batched_{bottom_up_fn_name}", jax.jit(bacthed_bottom_up_propagation_fn, static_argnames=bottom_up_inputs_names_optional[:-1])) # add batched version of the fonction


    top_down_fn_name = "top_down_propagation"
    top_down_inputs = [l for l in rbm_instance._neural_layers if l.priority == max(rbm_instance._neural_layers, key=lambda l: l.priority).priority]
    top_down_inputs += [l for l in rbm_instance._neural_layers if l not in rbm_instance._inferred_layers]
    top_down_inputs_names_mandatory = [l.name for l in top_down_inputs] 
    top_down_inputs_names_optional = ["K_steps", "K_meta_steps", "eps_stab", "verbose"]
    top_down_inputs_names = top_down_inputs_names_mandatory + top_down_inputs_names_optional

    newline = "\n"
    indent = "\t"
    docstring = f"""Computes a top-down propagation through the graph."""
    docstring += f"""{newline}Args:{newline+indent}{(newline+indent).join([f"{arg_name} : Array of same shape as {arg_name} layer, representing values for input vector of layer '{arg_name}'" for arg_name in top_down_inputs_names_mandatory])}"""
    docstring += f"""{newline+indent}K_steps (int) : Number of stabilization steps for inner graph stage stabilization if there are co-dependent layers. Default value to 5."""
    docstring += f"""{newline+indent}K_meta_steps (int) : Number of stabilization steps for hidden layers stack. Default value to 5."""
    docstring += f"""{newline+indent}eps_stab (float) : Minimum difference value between pre- and post-stabilization steps layers states used for early stopping stabilization. Default value to 1e-10."""
    docstring += f"""{newline+indent}verbose (bool) : Set verbosity of the function mainly for debug purposes and see the propagation process. Default value to False.{newline}"""
    docstring += f"""{newline}Returns:{newline+indent} dict[dict] : A dictionnary of layers states containning probabilities and sampled states. 
                      The dictionnary takes structure : dict('layer_name': dict('probabilities':probs, 'samples':sampled_states))"""

    def top_down_propagation_fn(*args, K_steps=5, K_meta_steps=5, eps_stab=1e-10, verbose=False, **kwargs):
        # check inputs
        fn_args = {k:None for k in top_down_inputs_names}
        fn_args['K_steps'] = K_steps
        fn_args['K_meta_steps'] = K_meta_steps
        fn_args['eps_stab'] = eps_stab
        fn_args['verbose'] = verbose
        if len(args) > 0:
            for i,arg in enumerate(args):
                fn_args[top_down_inputs_names[i]] = arg

        if len([*kwargs]) > 0:
            for k,v in kwargs.items():
                fn_args[k] = v

        raise_error = False
        for k,v in fn_args.items():
            try:
                if v.all() == None:
                    raise_error = True
                    break
            except:
                if v == None:
                    raise_error =True
                    break
        if raise_error:
            raise TypeError(f"{top_down_fn_name}() missing required positional argument '{k}'")
        # top down propagation
        sorted_layers = sorted(rbm_instance._neural_layers, key=lambda l: l.priority, reverse=True)
        highest_priority = max(sorted_layers, key=lambda l: l.priority).priority
        priorities = list(set([highest_priority-l.priority for l in sorted_layers]))
        priorities_dict = {k:list() for k in priorities}

        for l in sorted_layers:
            priorities_dict[highest_priority-l.priority].append(l)
        inputs = {name:fn_args[name] for name in top_down_inputs_names}
        states = stabilized_propagation(inputs, sorted_layers, priorities_dict, K_steps, K_meta_steps, eps_stab, verbose)
        states_dict = {l.name:{"probabilities":states[0][i], "samples":states[1][i]} for i,l in enumerate(sorted_layers)}
        return states_dict
    
    # Créer la signature de la fonction
    fn_signature = inspect.signature(top_down_propagation_fn)
    top_down_propagation_fn_args = []
    for arg in top_down_inputs_names_mandatory:
        param = inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        top_down_propagation_fn_args.append(param)

    for arg in fn_signature.parameters.keys():
        if arg not in ['args', 'kwargs']:
            default_value = fn_signature.parameters[arg].default
            param = inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default_value)
            top_down_propagation_fn_args.append(param)
    top_down_propagation_fn_signature = inspect.Signature(parameters=top_down_propagation_fn_args)
    top_down_propagation_fn.__signature__ = top_down_propagation_fn_signature
    top_down_propagation_fn.__doc__ = docstring
    
    setattr(rbm_instance, top_down_fn_name, jax.jit(top_down_propagation_fn, static_argnames=top_down_inputs_names_optional))

    def bacthed_top_down_propagation_fn(*args, K_steps=5, K_meta_steps=5, eps_stab=1e-10, **kwargs):
        fn_args = {k:None for k in top_down_inputs_names}
        fn_args['K_steps'] = K_steps
        fn_args['K_meta_steps'] = K_meta_steps
        fn_args['eps_stab'] = eps_stab
        fn_args['verbose'] = False
        if len(args) > 0:
            for i,arg in enumerate(args):
                fn_args[top_down_inputs_names[i]] = arg

        if len([*kwargs]) > 0:
            for k,v in kwargs.items():
                fn_args[k] = v


        raise_error = False
        for k,v in fn_args.items():
            try:
                if v.all() == None:
                    raise_error = True
                    break
            except:
                if v == None:
                    raise_error =True
                    break
        if raise_error:
            raise TypeError(f"batched_{top_down_fn_name}() missing required positional argument '{k}'")
        
        partial_fn = partial(getattr(rbm_instance,top_down_fn_name), 
                             K_steps=fn_args['K_steps'], K_meta_steps=fn_args['K_meta_steps'], 
                             eps_stab=fn_args['eps_stab'], verbose=fn_args['verbose'])
        return jax.vmap(partial_fn, in_axes=[0,0])(*[fn_args[a] for a in top_down_inputs_names_mandatory])

    bacthed_top_down_propagation_fn.__signature__ = top_down_propagation_fn_signature
    bacthed_top_down_propagation_fn.__doc__ = f"""** Batched version of {top_down_fn_name} using partial vmap **{newline+newline}"""+docstring

    setattr(rbm_instance, f"batched_{top_down_fn_name}", jax.jit(bacthed_top_down_propagation_fn, static_argnames=top_down_inputs_names_optional[:-1])) # add batched version of the fonction


    get_positive_states_fn_name = "get_positive_states"

    newline = "\n"
    indent = "\t"
    docstring = f"""Computes positives states through a bottom-up propagation."""
    docstring += f"""{newline}Args:{newline+indent}{(newline+indent).join([f"{arg_name} : Array of same shape as {arg_name} layer, representing values for input vector of layer '{arg_name}'" for arg_name in bottom_up_inputs_names_mandatory])}"""
    docstring += f"""{newline+indent}K_steps (int) : Number of stabilization steps for inner graph stage stabilization if there are co-dependent layers. Default value to 5."""
    docstring += f"""{newline+indent}K_meta_steps (int) : Number of stabilization steps for hidden layers stack. Default value to 5."""
    docstring += f"""{newline+indent}eps_stab (float) : Minimum difference value between pre- and post-stabilization steps layers states used for early stopping stabilization. Default value to 1e-10."""
    docstring += f"""{newline+indent}verbose (bool) : Set verbosity of the function mainly for debug purposes and see the propagation process. Default value to False.{newline}"""
    docstring += f"""{newline}Returns:{newline+indent} dict[dict] : A dictionnary of layers states containning probabilities and sampled states. 
                      The dictionnary takes structure : dict('layer_name': dict('probabilities':probs, 'samples':sampled_states))"""

    def get_positive_states_fn(*args, K_steps=5, K_meta_steps=5, eps_stab=1e-10, **kwargs):
        # check inputs
        fn_args = {k:None for k in bottom_up_inputs_names}
        fn_args['K_steps'] = K_steps
        fn_args['K_meta_steps'] = K_meta_steps
        fn_args['eps_stab'] = eps_stab
        fn_args['verbose'] = False
        if len(args) > 0:
            for i,arg in enumerate(args):
                fn_args[bottom_up_inputs_names[i]] = arg

        if len([*kwargs]) > 0:
            for k,v in kwargs.items():
                fn_args[k] = v

        raise_error = False
        for k,v in fn_args.items():
            try:
                if v.all() == None:
                    raise_error = True
                    break
            except:
                if v == None:
                    raise_error =True
                    break
        if raise_error:
            raise TypeError(f"{get_positive_states_fn_name}() missing required positional argument '{k}'")
        
        bottom_up_fn = getattr(rbm_instance, bottom_up_fn_name)
        bottom_up_fn_signature = inspect.signature(bottom_up_fn)
        bottom_up_fn_args_names = list(bottom_up_fn_signature.parameters.keys())

        return bottom_up_fn(*list(map(fn_args.get, bottom_up_fn_args_names)))

    # Créer la signature de la fonction
    fn_signature = inspect.signature(get_positive_states_fn)
    get_positive_states_fn_args = []
    for arg in bottom_up_inputs_names_mandatory:
        param = inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        get_positive_states_fn_args.append(param)

    for arg in fn_signature.parameters.keys():
        if arg not in ['args', 'kwargs']:
            default_value = fn_signature.parameters[arg].default
            param = inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default_value)
            get_positive_states_fn_args.append(param)
    get_positive_states_fn_signature = inspect.Signature(parameters=get_positive_states_fn_args)
    get_positive_states_fn.__signature__ = get_positive_states_fn_signature
    get_positive_states_fn.__doc__ = docstring
    
    setattr(rbm_instance, get_positive_states_fn_name, jax.jit(get_positive_states_fn, static_argnames=bottom_up_inputs_names_optional[:-1]))

    def bacthed_get_positive_states_fn(*args, K_steps=5, K_meta_steps=5, eps_stab=1e-10, **kwargs):
        fn_args = {k:None for k in bottom_up_inputs_names}
        fn_args['K_steps'] = K_steps
        fn_args['K_meta_steps'] = K_meta_steps
        fn_args['eps_stab'] = eps_stab
        fn_args['verbose'] = False
        if len(args) > 0:
            for i,arg in enumerate(args):
                fn_args[bottom_up_inputs_names[i]] = arg

        if len([*kwargs]) > 0:
            for k,v in kwargs.items():
                fn_args[k] = v


        raise_error = False
        for k,v in fn_args.items():
            try:
                if v.all() == None:
                    raise_error = True
                    break
            except:
                if v == None:
                    raise_error =True
                    break
        if raise_error:
            raise TypeError(f"batched_{get_positive_states_fn_name}() missing required positional argument '{k}'")
        
        partial_fn = partial(getattr(rbm_instance,get_positive_states_fn_name), 
                             K_steps=fn_args['K_steps'], K_meta_steps=fn_args['K_meta_steps'], 
                             eps_stab=fn_args['eps_stab'], verbose=fn_args['verbose'])
        return jax.vmap(partial_fn, in_axes=[0,0])(*[fn_args[a] for a in bottom_up_inputs_names_mandatory])

    bacthed_get_positive_states_fn.__signature__ = get_positive_states_fn_signature
    bacthed_get_positive_states_fn.__doc__ = f"""** Batched version of {get_positive_states_fn_name} using partial vmap **{newline+newline}"""+docstring

    setattr(rbm_instance, f"batched_{get_positive_states_fn_name}", jax.jit(bacthed_get_positive_states_fn, static_argnames=bottom_up_inputs_names_optional[:-1])) # add batched version of the fonction


    get_negative_states_fn_name = "get_negative_states"

    newline = "\n"
    indent = "\t"
    docstring = f"""Computes negative states through top-down then bottom-up propagations."""
    docstring += f"""{newline}Args:{newline+indent}{(newline+indent).join([f"{arg_name} : Array of same shape as {arg_name} layer, representing values for input vector of layer '{arg_name}'" for arg_name in top_down_inputs_names_mandatory])}"""
    docstring += f"""{newline+indent}K_steps (int) : Number of stabilization steps for inner graph stage stabilization if there are co-dependent layers. Default value to 5."""
    docstring += f"""{newline+indent}K_meta_steps (int) : Number of stabilization steps for hidden layers stack. Default value to 5."""
    docstring += f"""{newline+indent}eps_stab (float) : Minimum difference value between pre- and post-stabilization steps layers states used for early stopping stabilization. Default value to 1e-10."""
    docstring += f"""{newline+indent}verbose (bool) : Set verbosity of the function mainly for debug purposes and see the propagation process. Default value to False.{newline}"""
    docstring += f"""{newline}Returns:{newline+indent} dict[dict] : A dictionnary of layers states containning probabilities and sampled states. 
                      The dictionnary takes structure : dict('layer_name': dict('probabilities':probs, 'samples':sampled_states))"""


    def get_negative_states_fn(*args, K_steps=5, K_meta_steps=5, eps_stab=1e-10, **kwargs):
        # check inputs
        fn_args = {k:None for k in top_down_inputs_names}
        fn_args['K_steps'] = K_steps
        fn_args['K_meta_steps'] = K_meta_steps
        fn_args['eps_stab'] = eps_stab
        fn_args['verbose'] = False
        if len(args) > 0:
            for i,arg in enumerate(args):
                fn_args[top_down_inputs_names[i]] = arg

        if len([*kwargs]) > 0:
            for k,v in kwargs.items():
                fn_args[k] = v

        raise_error = False
        for k,v in fn_args.items():
            try:
                if v.all() == None:
                    raise_error = True
                    break
            except:
                if v == None:
                    raise_error =True
                    break
        if raise_error:
            raise TypeError(f"{get_negative_states_fn_name}() missing required positional argument '{k}'")
        
        top_down_fn = getattr(rbm_instance, top_down_fn_name)
        top_down_fn_signature = inspect.signature(top_down_fn)
        top_down_fn_args_names = list(top_down_fn_signature.parameters.keys())

        preneg_states = top_down_fn(*list(map(fn_args.get, top_down_fn_args_names)))

        bottom_up_fn = getattr(rbm_instance, bottom_up_fn_name)
        bottom_up_fn_signature = inspect.signature(bottom_up_fn)
        bottom_up_fn_args_names = list(bottom_up_fn_signature.parameters.keys())
        bottom_up_fn_neccessary_args_names = [param.name for param in bottom_up_fn_signature.parameters.values() if param.default == inspect.Parameter.empty]

        negstates_fn_args = fn_args.copy()
        for arg_name in bottom_up_fn_neccessary_args_names:
            negstates_fn_args[arg_name] = preneg_states[arg_name]["samples"]

        negative_states = bottom_up_fn(*list(map(negstates_fn_args.get, bottom_up_fn_args_names)))

        return negative_states
    
    # Créer la signature de la fonction
    fn_signature = inspect.signature(get_negative_states_fn)
    get_negative_states_fn_args = []
    for arg in top_down_inputs_names_mandatory:
        param = inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        get_negative_states_fn_args.append(param)

    for arg in fn_signature.parameters.keys():
        if arg not in ['args', 'kwargs']:
            default_value = fn_signature.parameters[arg].default
            param = inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default_value)
            get_negative_states_fn_args.append(param)
    get_negative_states_fn_signature = inspect.Signature(parameters=get_negative_states_fn_args)
    get_negative_states_fn.__signature__ = get_negative_states_fn_signature
    get_negative_states_fn.__doc__ = docstring
    
    setattr(rbm_instance, get_negative_states_fn_name, jax.jit(get_negative_states_fn, static_argnames=top_down_inputs_names_optional[:-1]))

    def bacthed_get_negative_states_fn(*args, K_steps=5, K_meta_steps=5, eps_stab=1e-10, **kwargs):
        fn_args = {k:None for k in top_down_inputs_names}
        fn_args['K_steps'] = K_steps
        fn_args['K_meta_steps'] = K_meta_steps
        fn_args['eps_stab'] = eps_stab
        fn_args['verbose'] = False
        if len(args) > 0:
            for i,arg in enumerate(args):
                fn_args[top_down_inputs_names[i]] = arg

        if len([*kwargs]) > 0:
            for k,v in kwargs.items():
                fn_args[k] = v


        raise_error = False
        for k,v in fn_args.items():
            try:
                if v.all() == None:
                    raise_error = True
                    break
            except:
                if v == None:
                    raise_error =True
                    break
        if raise_error:
            raise TypeError(f"batched_{get_negative_states_fn_name}() missing required positional argument '{k}'")
        
        partial_fn = partial(getattr(rbm_instance,get_negative_states_fn_name), 
                             K_steps=fn_args['K_steps'], K_meta_steps=fn_args['K_meta_steps'], 
                             eps_stab=fn_args['eps_stab'], verbose=fn_args['verbose'])
        return jax.vmap(partial_fn, in_axes=[0,0])(*[fn_args[a] for a in top_down_inputs_names_mandatory])

    bacthed_get_negative_states_fn.__signature__ = get_negative_states_fn_signature
    bacthed_get_negative_states_fn.__doc__ = f"""** Batched version of {get_negative_states_fn_name} using partial vmap **{newline+newline}"""+docstring

    setattr(rbm_instance, f"batched_{get_negative_states_fn_name}", jax.jit(bacthed_get_negative_states_fn, static_argnames=top_down_inputs_names_optional[:-1])) # add batched version of the fonction
        
def _create_compute_grads_fns(rbm_instance):
    grads_fn_name = "compute_grads"

    grads_inputs_names = ["positive_states", "negative_states"]
    neural_layers_names = [l.name for l in rbm_instance._neural_layers]

    def compute_grads_fn(*args, **kwargs):
        # check inputs
        fn_args = {k:None for k in grads_inputs_names}
        if len(args) > 0:
            for i,arg in enumerate(args):
                fn_args[grads_inputs_names[i]] = arg

        if len([*kwargs]) > 0:
            for k,v in kwargs.items():
                fn_args[k] = v

        raise_error = False
        for k,v in fn_args.items():
            try:
                if v.all() == None:
                    raise_error = True
                    break
            except:
                if v == None:
                    raise_error =True
                    break
        if raise_error: raise TypeError(f"{grads_fn_name}() missing required positional argument '{k}'")
        
        pos_states = fn_args[grads_inputs_names[0]]
        neg_states = fn_args[grads_inputs_names[1]]
        
        try: 
            assert isinstance(pos_states, dict)
            try: assert sorted(pos_states.keys()) == sorted(neural_layers_names)
            except: raise ValueError(f"{grads_inputs_names[0]} must have keys : {neural_layers_names}")
        except: raise ValueError(f"{grads_inputs_names[0]} must be type {dict.__name__}")

        try: 
            assert isinstance(neg_states, dict)
            try: assert sorted(neg_states.keys()) == sorted(neural_layers_names)
            except: raise ValueError(f"{grads_inputs_names[1]} must have keys : {neural_layers_names}")
        except: raise ValueError(f"{grads_inputs_names[1]} must be type {dict.__name__}")
        
        get_energy_grads_fn = getattr(rbm_instance, "get_energy_grad")

        positive_energy_grads = get_energy_grads_fn(*[pos_states[lname]['probabilities'] for lname in neural_layers_names])
        negative_energy_grads = get_energy_grads_fn(*[neg_states[lname]['probabilities'] for lname in neural_layers_names])

        grads = dict()
        for pgrads, ngrads in zip(positive_energy_grads,negative_energy_grads):
            for (pk,pv),(nk,nv) in zip(pgrads.items(),ngrads.items()):
                assert pk == nk
                grads[pk] = pv - nv

        return grads
    
    def batched_compute_grads_fn(*args, **kwargs):
        # check inputs
        fn_args = {k:None for k in grads_inputs_names}
        if len(args) > 0:
            for i,arg in enumerate(args):
                fn_args[grads_inputs_names[i]] = arg

        if len([*kwargs]) > 0:
            for k,v in kwargs.items():
                fn_args[k] = v

        raise_error = False
        for k,v in fn_args.items():
            try:
                if v.all() == None:
                    raise_error = True
                    break
            except:
                if v == None:
                    raise_error =True
                    break
        if raise_error: raise TypeError(f"batched_{grads_fn_name}() missing required positional argument '{k}'")

        pos_states = fn_args[grads_inputs_names[0]]
        neg_states = fn_args[grads_inputs_names[1]]

        try: 
            assert isinstance(pos_states, dict)
            try: assert sorted(pos_states.keys()) == sorted(neural_layers_names)
            except: raise ValueError(f"{grads_inputs_names[0]} must have keys : {neural_layers_names}")
        except: raise ValueError(f"{grads_inputs_names[0]} must be type {dict.__name__}")

        try: 
            assert isinstance(neg_states, dict)
            try: assert sorted(neg_states.keys()) == sorted(neural_layers_names)
            except: raise ValueError(f"{grads_inputs_names[1]} must have keys : {neural_layers_names}")
        except: raise ValueError(f"{grads_inputs_names[1]} must be type {dict.__name__}")
        
        get_energy_grads_fn = getattr(rbm_instance, "batched_get_energy_grad")

        positive_energy_grads = get_energy_grads_fn(*[pos_states[lname]['probabilities'] for lname in neural_layers_names])
        negative_energy_grads = get_energy_grads_fn(*[neg_states[lname]['probabilities'] for lname in neural_layers_names])

        grads = dict()
        for pgrads, ngrads in zip(positive_energy_grads,negative_energy_grads):
            for (pk,pv),(nk,nv) in zip(pgrads.items(),ngrads.items()):
                assert pk == nk
                grads[pk] = pv - nv

        return grads

    cdk_fn_name = "CD_k"
    cdk_inputs = [l for l in rbm_instance._neural_layers if l.priority == min(rbm_instance._neural_layers, key=lambda l: l.priority).priority]
    cdk_inputs_names_mandatory = [l.name for l in cdk_inputs]
    cdk_inputs_names_optional = ["K", "Nb_stabilization_steps", "espsilon_stabilization"]
    cdk_inputs_names = cdk_inputs_names_mandatory+cdk_inputs_names_optional

    def CD_k_fn(*args, K=5, Nb_stabilization_steps=5, espsilon_stabilization=1e-10, **kwargs):
        # check inputs
        fn_args = {k:None for k in cdk_inputs_names}
        fn_args['K'] = K
        fn_args['Nb_stabilization_steps'] = Nb_stabilization_steps
        fn_args['espsilon_stabilization'] = espsilon_stabilization
        if len(args) > 0:
            for i,arg in enumerate(args):
                fn_args[cdk_inputs_names[i]] = arg

        if len([*kwargs]) > 0:
            for k,v in kwargs.items():
                fn_args[k] = v


        raise_error = False
        for k,v in fn_args.items():
            try:
                if v.all() == None:
                    raise_error = True
                    break
            except:
                if v == None:
                    raise_error =True
                    break
        if raise_error: raise TypeError(f"{cdk_fn_name}() missing required positional argument '{k}'")

        get_positive_states_fn = getattr(rbm_instance, "get_positive_states")
        get_negative_states_fn = getattr(rbm_instance, "get_negative_states")

        positive_states = get_positive_states_fn(*([fn_args[arg_name] for arg_name in cdk_inputs_names]))
        get_negative_states_fn_args = inspect.signature(get_negative_states_fn).parameters
        get_negative_states_fn_positional_args = [param for param in get_negative_states_fn_args.values() if param.default is inspect.Parameter.empty]
        negative_states = get_negative_states_fn(*([positive_states[arg.name]['probabilities'] for arg in get_negative_states_fn_positional_args] +\
                                                   [fn_args[arg_name] for arg_name in cdk_inputs_names_optional]))
        
        return compute_grads_fn(positive_states, negative_states)
    
    def batched_CD_k_fn(*args, K=5, Nb_stabilization_steps=5, espsilon_stabilization=1e-10, **kwargs):
        # check inputs
        fn_args = {k:None for k in cdk_inputs_names}
        fn_args['K'] = K
        fn_args['Nb_stabilization_steps'] = Nb_stabilization_steps
        fn_args['espsilon_stabilization'] = espsilon_stabilization
        if len(args) > 0:
            for i,arg in enumerate(args):
                fn_args[cdk_inputs_names[i]] = arg

        if len([*kwargs]) > 0:
            for k,v in kwargs.items():
                fn_args[k] = v


        raise_error = False
        for k,v in fn_args.items():
            try:
                if v.all() == None:
                    raise_error = True
                    break
            except:
                if v == None:
                    raise_error =True
                    break
        if raise_error: raise TypeError(f"batched_{cdk_fn_name}() missing required positional argument '{k}'")

        get_positive_states_fn = getattr(rbm_instance, "batched_get_positive_states")
        get_negative_states_fn = getattr(rbm_instance, "batched_get_negative_states")

        positive_states = get_positive_states_fn(*([fn_args[arg_name] for arg_name in cdk_inputs_names]))
        get_negative_states_fn_args = inspect.signature(get_negative_states_fn).parameters
        get_negative_states_fn_positional_args = [param for param in get_negative_states_fn_args.values() if param.default is inspect.Parameter.empty]
        negative_states = get_negative_states_fn(*([positive_states[arg.name]['probabilities'] for arg in get_negative_states_fn_positional_args] +\
                                                   [fn_args[arg_name] for arg_name in cdk_inputs_names_optional]))
        
        return batched_compute_grads_fn(positive_states, negative_states)

    newline = "\n"
    indent = "\t"
    docstring = f"""Computes gradients given positive and negative states"""
    docstring += f"""{newline}Args:{newline+indent}{(newline+indent).join([f"{arg_name} : Dictionary of dictionaries representing {' '.join(arg_name.split('_'))}. The dictionnary takes structure : dict('layer_name': dict('probabilities':probs, 'samples':sampled_states))" for arg_name in grads_inputs_names])}"""
    docstring += f"""{newline}Returns:{newline+indent} dict : A dictionnary of gradients for each trainable parameters. The dictionnary takes structure : dict('parameter_name':gradient_value)."""
    fn_signature = inspect.signature(compute_grads_fn)
    fn_args = []
    for arg in grads_inputs_names:
        param = inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        fn_args.append(param)

    for arg in fn_signature.parameters.keys():
        if arg not in ['args', 'kwargs']:
            default_value = fn_signature.parameters[arg].default
            param = inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default_value)
            fn_args.append(param)
    fn_signature = inspect.Signature(parameters=fn_args)
    compute_grads_fn.__signature__ = fn_signature
    compute_grads_fn.__doc__ = docstring
    setattr(rbm_instance, grads_fn_name, jax.jit(compute_grads_fn))

    batched_compute_grads_fn.__signature__ = fn_signature
    batched_compute_grads_fn.__doc__ = f"""** Batched version of {grads_fn_name}**{newline+newline}"""+docstring
    setattr(rbm_instance, f"batched_{grads_fn_name}", jax.jit(batched_compute_grads_fn))

    fn_signature = inspect.signature(CD_k_fn)
    fn_args = []
    for arg in cdk_inputs_names_mandatory:
        param = inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        fn_args.append(param)
    for arg in fn_signature.parameters.keys():
        if arg not in ['args', 'kwargs']:
            default_value = fn_signature.parameters[arg].default
            param = inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default_value)
            fn_args.append(param)
    fn_signature = inspect.Signature(parameters=fn_args)
    CD_k_fn.__signature__ = fn_signature
    docstring = f"""Computes CD-k gradients given graph inputs"""
    docstring += f"""{newline}Args:{newline+indent}{(newline+indent).join([f"{arg_name} : Array of same shape as {arg_name} layer, representing values for input vector of layer '{arg_name}'" for arg_name in cdk_inputs_names_mandatory])}"""    
    docstring += f"""{newline+indent}K (int) : Number of stabilization steps for inner graph stage stabilization if there are co-dependent layers. 
                     Default value {fn_signature.parameters["K"]}."""
    docstring += f"""{newline+indent}Nb_stabilization_steps (int) : Number of stabilization steps for hidden layers stack. 
                     Default value {fn_signature.parameters["Nb_stabilization_steps"]}."""
    docstring += f"""{newline+indent}espsilon_stabilization (float) : Minimum difference value between pre- and post-stabilization steps layers states used for early stopping stabilization. 
                     Default value to {fn_signature.parameters["espsilon_stabilization"]}."""
    docstring += f"""{newline}Returns:{newline+indent} dict : A dictionnary of gradients for each trainable parameters. The dictionnary takes structure : dict('parameter_name':gradient_value)."""
    CD_k_fn.__doc__ = docstring
    setattr(rbm_instance, cdk_fn_name, jax.jit(CD_k_fn, static_argnames=cdk_inputs_names_optional))

    batched_CD_k_fn.__signature__ = fn_signature
    batched_CD_k_fn.__doc__ = f"""** Batched version of {cdk_fn_name}**{newline+newline}""" + docstring
    setattr(rbm_instance, f"batched_{cdk_fn_name}", jax.jit(batched_CD_k_fn, static_argnames=cdk_inputs_names_optional))

def _create_train_and_test_fns(rbm_instance):

    apply_grads_fn_name = "apply_gradients"

    def apply_grads_fn(gradients_dict, learning_rate):
        for prm_name in gradients_dict.keys():
            prm = getattr(rbm_instance, prm_name)
            new_prm = prm - learning_rate * gradients_dict[prm_name]
            setattr(rbm_instance, prm_name, new_prm)
            if prm_name.startwith('W_'): dict_name = "weights_dict"
            elif prm_name.startwith('B_'): dict_name = "biases_dict"
            elif prm_name.startwith('S_'): dict_name = "sigmas_dict"
            prm_dict = getattr(rbm_instance, prm_dict)
            prm_dict[prm_name] = new_prm
            setattr(rbm_instance, dict_name, prm_dict)

    setattr(rbm_instance, apply_grads_fn_name, jax.jit(apply_grads_fn))

    train_fn_name = "train"
    train_inputs = [l for l in rbm_instance._neural_layers if l.priority == min(rbm_instance._neural_layers, key=lambda l: l.priority).priority]
    train_inputs_names_mandatory = [l.name for l in train_inputs]
    train_inputs_names_optional = ["K", "Nb_stabilization_steps", "espsilon_stabilization", "learning_rate"]
    train_inputs_names = train_inputs_names_mandatory+train_inputs_names_optional

    def train_fn(*args, K=5, Nb_stabilization_steps=5, espsilon_stabilization=1e-10, learning_rate=1e-3, **kwargs):
        # check inputs
        fn_args = {k:None for k in train_inputs_names}
        fn_args['K'] = K
        fn_args['Nb_stabilization_steps'] = Nb_stabilization_steps
        fn_args['espsilon_stabilization'] = espsilon_stabilization
        fn_args['learning_rate'] = learning_rate
        if len(args) > 0:
            for i,arg in enumerate(args):
                fn_args[train_inputs_names[i]] = arg

        if len([*kwargs]) > 0:
            for k,v in kwargs.items():
                fn_args[k] = v


        raise_error = False
        for k,v in fn_args.items():
            try:
                if v.all() == None:
                    raise_error = True
                    break
            except:
                if v == None:
                    raise_error =True
                    break
        if raise_error: raise TypeError(f"batched_{train_fn_name}() missing required positional argument '{k}'")

        get_cdk_grads = getattr(rbm_instance, "batched_CD_k")
        cdk_grads = get_cdk_grads(*[fn_args[arg_name] for arg_name in train_inputs_names if arg_name != "learning_rate"])

        apply_grads = getattr(rbm_instance, "apply_gradients")
        apply_grads(cdk_grads, learning_rate)

    setattr(rbm_instance, train_fn_name, jax.jit(train_fn, static_argnames=train_inputs_names_optional))

    test_fn_name = "test"
    test_inputs = [l for l in rbm_instance._neural_layers if l.priority == min(rbm_instance._neural_layers, key=lambda l: l.priority).priority]
    test_inputs_names_mandatory = [l.name for l in test_inputs] + ["error_fn"]
    test_inputs_names_optional = ["K", "Nb_stabilization_steps", "espsilon_stabilization"]
    test_inputs_names = test_inputs_names_mandatory+test_inputs_names_optional

    def test_fn(*args, K=5, Nb_stabilization_steps=5, espsilon_stabilization=1e-10, **kwargs):
        # check inputs
        fn_args = {k:None for k in test_inputs_names}
        fn_args['K'] = K
        fn_args['Nb_stabilization_steps'] = Nb_stabilization_steps
        fn_args['espsilon_stabilization'] = espsilon_stabilization
        if len(args) > 0:
            for i,arg in enumerate(args):
                fn_args[test_inputs_names[i]] = arg

        if len([*kwargs]) > 0:
            for k,v in kwargs.items():
                fn_args[k] = v


        raise_error = False
        for k,v in fn_args.items():
            try:
                if v.all() == None:
                    raise_error = True
                    break
            except:
                if v == None:
                    raise_error =True
                    break
        if raise_error: raise TypeError(f"batched_{test_fn_name}() missing required positional argument '{k}'")

        get_positive_states_fn = getattr(rbm_instance, "batched_get_positive_states")
        get_negative_states_fn = getattr(rbm_instance, "batched_get_negative_states")

        positive_states = get_positive_states_fn(*([fn_args[arg.name] for arg in test_inputs]+[fn_args[arg_name] for arg_name in test_inputs_names_optional]))
        get_negative_states_fn_args = inspect.signature(get_negative_states_fn).parameters
        get_negative_states_fn_positional_args = [param for param in get_negative_states_fn_args.values() if param.default is inspect.Parameter.empty]
        negative_states = get_negative_states_fn(*([positive_states[arg.name]['probabilities'] for arg in get_negative_states_fn_positional_args] +\
                                                   [fn_args[arg_name] for arg_name in test_inputs_names_optional]))
        
        visibles = test_inputs.copy()
        inferred_visibles = [visible for visible in visibles if visible in getattr(rbm_instance, "_inferred_layers")]

        positive_inferred_visibles = [positive_states[l.name] for l in inferred_visibles]
        negative_inferred_visibles = [negative_states[l.name] for l in inferred_visibles]

        error = fn_args["error_fn"](positive_inferred_visibles, negative_inferred_visibles)

        return error
    
    setattr(rbm_instance, test_fn_name, jax.jit(test_fn, static_argnames=["error_fn"]+test_inputs_names_optional))


def _create_instance_functions(rbm_instance):
    _create_factors_fns(rbm_instance)
    _create_get_probs_fns(rbm_instance)
    _create_samples_fns(rbm_instance)
    _create_get_energy_fn(rbm_instance)
    _create_propagation_fns(rbm_instance)
    _create_compute_grads_fns(rbm_instance)
    _create_train_and_test_fns(rbm_instance)
    



