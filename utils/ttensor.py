import theano,theano.tensor as T

def find_inputs_and_params(node):
    '''Walk a computation graph and extract root variables.

    Parameters
    ----------
    node : Theano expression
        A symbolic Theano expression to walk.

    Returns
    -------
    inputs : list Theano variables
        A list of candidate inputs for this graph. Inputs are nodes in the graph
        with no parents that are not shared and are not constants.
    params : list of Theano shared variables
        A list of candidate parameters for this graph. Parameters are nodes in
        the graph that are shared variables.
    '''
    queue, seen, inputs, params = [node], set(), set(), set()
    while queue:
        node = queue.pop()
        seen.add(node)
        queue.extend(p for p in node.get_parents() if p not in seen)
        if not node.get_parents():
            if isinstance(node, theano.compile.SharedVariable):
                params.add(node)
            elif not isinstance(node, T.Constant):
                inputs.add(node)
    return list(inputs), list(params)

