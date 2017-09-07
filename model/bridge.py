import nn
import ops
import theano,theano.tensor as T

def map_key(states, dim_state, dim_key):
    key = nn.linear(states, [dim_state, dim_key], False, scope="map-key")
    return key


def attention(query, keys, key_mask, dim_query, dim_key, dtype=None, scope=None):
    with ops.variable_scope(scope or "attention", dtype=dtype):
        # content-based addressing
        # e_i = v_a^T tanh(W query + key_i)
        # alpha = softmax({e_i})
        # (n_query, dim_query) -> (n_query, dim_key)
        mapped_query = nn.linear(query, [dim_query, dim_key], False, scope="map-query")
        # (n_key, n_query, dim_key)
        act = T.tanh(mapped_query[None, :, :] + keys)
        # (n_key, n_query, 1)
        e = nn.linear(act, [dim_key, 1], False, scope="pre-alpha")  # (n_key, n_query, 1)
        # (n_key, n_query)
        e = T.reshape(e, e.shape[:2])
        e = e.T  # (n_query, n_key)
        # match dimension
        key_mask=key_mask.T
        alpha = nn.masked_softmax(e, key_mask)  # (n_query, n_key)
        alpha = alpha.T  # (n_key, n_query)
    return alpha